import numpy as np
import torch, yaml
from sklearn.neighbors._kd_tree import KDTree
from torch.utils.data import Dataset
from os import path
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SemanticKITTIDataset(Dataset):
    name = "SemanticKITTI"

    def __init__(self, root_path="../semantic_kitti",
                 splits_path="data/SemanticKITTI",
                 split="train",
                 CL=False,
                 step=0,
                 inpaint=False,
                 setup="Sequential_masked",
                 method="CIL",
                 c2f=False,
                 ues=False,
                 return_ref=False,
                 augment=True):

        self.ues = ues
        self.root_path = root_path
        self.splits_path = splits_path
        self.augment = augment
        self.return_ref = return_ref

        self.split = split
        self.step = step
        self.method = method
        self.setup = setup
        if setup == "Coarse-to-fine":
            c2f = True
        self.c2f = c2f
        self.inpaint = inpaint

        label_mapping = path.join(splits_path, 'semantic-kitti.yaml')
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.idmap = self.init_idmap()

        if CL:
            CL_mapping = path.join(splits_path, method, 'CILconfig.yaml') #CIL
            with open(CL_mapping, 'r') as stream:
                CLyaml = yaml.safe_load(stream)
            self.step_map, self.idmap = self.init_step_map(CLyaml)
        else:
            self.step_map = np.arange(0, len(self.idmap))

        self.cmap = self.init_cmap()
        self.weights = self.init_weights()
        self.cnames = list(self.idmap.keys())

        if split == 'train' and CL == True and setup != "Overlapped":
            self.items = [l.strip() for l in
                          open(path.join(splits_path, method, "step_" + str(step) + '.txt'), 'r')]  # _backup
        else:
            self.items = [l.strip() for l in open(path.join(splits_path, split + '.txt'), 'r')]

    def init_step_map(self, CLyaml):
        if self.c2f:
            # if coarse to fine
            c2f_step_map = CLyaml['c2f']['step_' + str(self.step)]
            step_map = np.vectorize(c2f_step_map.__getitem__)(np.arange(0, len(c2f_step_map)))
            idmap = {k: v for k, v in self.idmap.items() if v in np.unique(step_map)}
            #idmap = {k: v for k, v in self.idmap.items() if (step_map[v] == v)}
        else:
            # if not coarse to fine
            label_to_step = CLyaml['label_to_step']
            if self.split == 'train' and (self.setup == 'Sequential_masked' or self.setup == "Disjoint" or self.setup == "Overlapped"):
                mask = (np.vectorize(label_to_step.__getitem__)(
                    np.arange(0, len(label_to_step)))) != self.step  # sequential masked
            else:
                mask = (np.vectorize(label_to_step.__getitem__)(
                    np.arange(0, len(label_to_step)))) > self.step  # sequential for val
            step_map = np.arange(0, len(label_to_step))
            step_map[mask] = 0
            idmap = {k: v for k, v in self.idmap.items() if (step_map[v] != 0 or v == 0)}

        return step_map, idmap

    def init_cmap(self):
        cmap = np.array([[0, 0, 0],  # unlabeled
                         [245, 150, 100],  # car
                         [245, 230, 100],  # bike
                         [150, 60, 30],  # motorcycle
                         [180, 30, 80],  # truck
                         [255, 0, 0],  # other-vehicle
                         [30, 30, 255],  # person
                         [200, 40, 255],  # bicyclist
                         [90, 30, 150],  # motorcyclist
                         [255, 0, 255],  # road
                         [255, 150, 255],  # parking
                         [75, 0, 75],  # sidewalk
                         [75, 0, 175],  # other-ground
                         [0, 200, 255],  # building
                         [50, 120, 255],  # fence
                         [0, 175, 0],  # vegetation
                         [0, 60, 135],  # trunck
                         [80, 240, 150],  # terrain
                         [150, 240, 255],  # pole
                         [0, 0, 255]], dtype=np.uint8)  # traffic sign
        return cmap

    def init_idmap(self):
        idmap = {0: 'unlabeled',
                 1: 'car',
                 2: 'bicycle',
                 3: 'motorcycle',
                 4: 'truck',
                 5: 'other-vehicle',
                 6: 'person',
                 7: 'bicyclist',
                 8: 'motorcyclist',
                 9: 'road',
                 10: 'parking',
                 11: 'sidewalk',
                 12: 'other-ground',
                 13: 'building',
                 14: 'fence',
                 15: 'vegetation',
                 16: 'trunk',
                 17: 'terrain',
                 18: 'pole',
                 19: 'traffic-sign'}
        idmap = {v: k for k, v in idmap.items()}
        return idmap

    def init_weights(self):  # calcolato su tutti i punti del training set: useless per CL
        num_per_class = np.array([1e30, 55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                  240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                  9833174, 129609852, 4506626, 1168181])
        weight = num_per_class / float(sum(num_per_class[1:]))
        ce_label_weight = 1 / (weight + 0.02)
        return ce_label_weight

    def __len__(self):
        return len(self.items)

    def color_label(self, lab, norm=True):
        if norm:
            return self.cmap[lab] / 255.
        else:
            return self.cmap[lab]

    def __getitem__(self, item):

        fname = path.join(self.root_path, self.items[item] + '.bin')

        if self.inpaint:
            lname = path.join('data', self.items[item].replace('dataset', 'semantic-kitti-inpaint').replace('velodyne',
                                                                                                            'pseudo_labels/in_painting') + '.label')
        else:
            lname = path.join(self.root_path, self.items[item].replace('velodyne', 'labels') + '.label')

        raw_data = np.fromfile(fname, dtype=np.float32).reshape((-1, 4))
        if self.split == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(lname, dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        xyz = np.array(raw_data[:, :3])
        lab = np.reshape(annotated_data.astype(np.uint8), newshape=(annotated_data.shape[0],))

        if self.setup == "Disjoint":
            CL_mapping = path.join(self.splits_path, self.method, 'CILconfig.yaml')  # CIL
            with open(CL_mapping, 'r') as stream:
                CLyaml = yaml.safe_load(stream)
            label_to_step = CLyaml["label_to_step"]
            mask = (np.vectorize(label_to_step.__getitem__)(
                    np.arange(0, len(label_to_step)))) > self.step-1  # sequential for val
            step_map = np.arange(0, len(label_to_step))
            step_map[mask] = 0
            mask = step_map != 0
            self.step_map[mask] = -1

        if not self.inpaint:
            lab = self.step_map[lab]  # map labels according to step

        if self.ues:
            lab, _, _ = unbiased_equalization_subsampling(xyz, lab,
                                                                     method="inner", #self.ues.method,
                                                                     threshold=1000, #self.ues.threshold,
                                                                     margin=1, #self.ues.margin,
                                                                     n_neighbors=16, #self.ues.n_neighbors
                                                                          )
            mask = lab>=0
            lab = lab[mask]
            xyz = xyz[mask, :]

        data_tuple = (torch.from_numpy(xyz).unsqueeze(0), torch.from_numpy(lab))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple

        """
        if self.augment and np.random.random()<.5:
            xyz += np.random.randn(*xyz.shape)

        # center & rescale PC in [-1,1]
        xyz -= xyz.mean(axis=0)
        xyz /= np.abs(xyz).max()

        if self.augment:
            # random rotation
            if np.random.random()<.5:
                r = R.from_rotvec(np.pi*(np.random.random(3,)-.5)*np.array([0.1,0.1,1])).as_matrix()
                xyz = np.einsum('jk,nj->nk',r,xyz)
                xyz /= np.abs(xyz).max()

            # random shift
            if np.random.random()<.5:
                xyz -= np.random.random((3,))*2-1.
            else:
                xyz += 1

            # random rescale & crop
            if np.random.random()<.5:
                if np.random.random()<.5:
                    xyz = np.round(xyz*(self.cube_edge//2)*np.random.random()).astype(int)
                else:
                    xyz = np.round(xyz*(self.cube_edge//2)/np.random.random()).astype(int)
            else:
                xyz = np.round(xyz*(self.cube_edge//2)).astype(int)

            valid = np.logical_and(np.all(xyz>-1, axis=1), np.all(xyz<self.cube_edge, axis=1))
        else:
            xyz += 1
            xyz = np.round(xyz*(self.cube_edge//2)).astype(int)
            valid = np.logical_and(np.all(xyz>-1, axis=1), np.all(xyz<self.cube_edge, axis=1))

        xyz = xyz[valid,:]
        lab = lab[valid]

        geom = np.zeros((self.cube_edge, self.cube_edge, self.cube_edge), dtype=np.float32)
        geom[tuple(xyz.T)] = 1

        labs = np.zeros((self.cube_edge, self.cube_edge, self.cube_edge), dtype=np.long)
        labs[tuple(xyz.T)] = lab

        return torch.from_numpy(geom).unsqueeze(0), torch.from_numpy(labs)
        """


#######################
#my sampling

def unbiased_equalization_subsampling(X, Y, method="random", threshold=1000, n_neighbors=16, margin=4):
    """
    subsampling function for point clouds
    @param method: defines the methodology to do subsampling
      'random': random subsampling
      'borders': compute neighbors n_neighbors and sample regions with minor number of neighbors of same class
      'inner': compute neighbors n_neighbors and sample regions with major number of neighbors of same class

      'threshold': border subsampling only if class frequency > threshold
      'mixed': threshold sampling and random subsampling only inner regions
    """
    classes = np.unique(Y)

    if method == "borders":
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        matrix = nbrs.kneighbors_graph(X).toarray()
        # confmat = np.zeros((len(classes), len(classes)))
        # for i,y in enumerate(Y):
        #  confmat[y, Y[indices]] = distances
        equal_neighbors = np.where(Y[indices] == Y[:, np.newaxis])[0]
        counts = np.bincount(equal_neighbors, minlength=Y.shape[0])
        Y_thresh = np.where(counts >= n_neighbors - margin, -1, Y)

    elif method == "inner":
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        matrix = nbrs.kneighbors_graph(X).toarray()
        # confmat = np.zeros((len(classes), len(classes)))
        # for i,y in enumerate(Y):
        #  confmat[y, Y[indices]] = distances
        equal_neighbors = np.where(Y[indices] == Y[:, np.newaxis])[0]
        counts = np.bincount(equal_neighbors, minlength=Y.shape[0])
        Y_thresh = np.where(counts < n_neighbors - margin, -1, Y)

    elif method == "random":
        Y_thresh = np.zeros_like(Y) - 1
        for c in classes:
            mask = np.stack(np.where(Y == c))[0]
            perm = np.random.permutation(mask)
            perm = perm[:threshold]
            Y_thresh[perm] = Y[perm]

    elif method == "threshold":
        # border subsampling
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        matrix = nbrs.kneighbors_graph(X).toarray()
        equal_neighbors = np.where(Y[indices] == Y[:, np.newaxis])[0]
        counts = np.bincount(equal_neighbors, minlength=Y.shape[0])
        Y_thresh = np.where(counts >= n_neighbors - margin, -1, Y)
        # re-add classes with few points
        for c in classes:
            mask = np.stack(np.where(Y == c))[0]
            if len(mask) < threshold:
                Y_thresh[mask] = Y[mask]

    elif method == "mixed":
        # border subsampling
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        matrix = nbrs.kneighbors_graph(X).toarray()
        equal_neighbors = np.where(Y[indices] == Y[:, np.newaxis])[0]
        counts = np.bincount(equal_neighbors, minlength=Y.shape[0])
        Y_thresh = np.where(counts >= n_neighbors - margin, -1, Y)
        # random subsampling
        for c in classes:
            mask = np.stack(np.where(Y == c))[0]
            perm = np.random.permutation(mask)
            perm = perm[:threshold]
            Y_thresh[perm] = Y[perm]

    cl, th = np.unique(Y_thresh[Y_thresh != -1], return_counts=True)
    return Y_thresh, cl, th

#######################
# for cylinder3d

class voxel_dataset(Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=-1, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, 50, 1.5], min_volume_space=[-50, -50, -3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        xyz = xyz.numpy().squeeze(axis=0)
        labels = labels.numpy().reshape((labels.shape[0], 1))

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        max_bound = np.percentile(xyz, 100, axis=0)
        min_bound = np.percentile(xyz, 0, axis=0)

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size

        intervals = crop_range / (cur_grid_size - 1)
        if (intervals == 0).any(): print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        data_tuple = (
        voxel_position, processed_label)  # (torch.from_numpy(voxel_position), torch.from_numpy(processed_label))

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels,
                           return_fea)  # (torch.from_numpy(grid_ind), torch.from_numpy(labels), torch.from_numpy(return_fea))
        return data_tuple


def gan_data_preprocessing(xyz, scale=200, quantization=[.01, .1, 2], interpolation=10):
    xyz = scale * xyz  # resize
    pc = np.zeros_like(xyz)
    # convert to polar coordinates
    for i, p in enumerate(xyz):
        pc[i] = cart2polar(p.reshape(1, -1))
    # homemade interpolation
    big_pc = []
    big_pc.append(pc)
    for index in np.arange(0, interpolation):
        new_pc = np.zeros_like(xyz)
        for i, p in enumerate(pc):
            new_pc[i][0] = p[0] + np.random.randn() / 1000
            new_pc[i][1] = p[1] + np.random.randn() / 1000
        big_pc.append(new_pc)
    pc = np.vstack(big_pc)
    # TODO: cosi è uniforme la quantizzazione, magari farla gaussiana
    for quantiz_step in quantization:
        # quantize over concentric circles
        for i, p in enumerate(pc):
            if np.round(p[0]) % quantiz_step <= quantiz_step / 2:
                pc[i][0] = p[0] - (np.random.randn() / 1000) * (quantiz_step / 2)
            elif np.round(p[0]) % quantiz_step > quantiz_step / 2:
                pc[i][0] = p[0] + (np.random.randn() / 1000) * (quantiz_step / 2)
            else:
                continue
    # quantize on z
    quantiz_step = 0.01
    for i, p in enumerate(pc):
        if np.round(p[2]) % (quantiz_step) <= quantiz_step / 2:
            pc[i][2] = - (p[2] - quantiz_step / 2)
        elif np.round(p[2]) % (quantiz_step) > quantiz_step / 2:
            pc[i][2] = - (p[2] + quantiz_step / 2)
        else:
            continue
    new_pc = np.zeros_like(pc)
    # convert to cartesian coordinates
    for i, p in enumerate(pc):
        new_pc[i] = polar2cart(p)
    return new_pc


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cart(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


class cylinder_dataset(Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=-1, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False,
                 transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4, max_rad=np.pi / 4):

        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.transform = transform_aug
        self.trans_std = trans_std

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def rotation_points_single_angle(self, points, angle, axis=0):
        # points: [N, 3]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if axis == 1:
            rot_mat_T = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype)
        elif axis == 2 or axis == -1:
            rot_mat_T = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype)
        elif axis == 0:
            rot_mat_T = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype)
        else:
            raise ValueError("axis should in range")

        return points @ rot_mat_T

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]

        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        xyz = xyz.numpy().squeeze(axis=0)
        labels = labels.numpy().reshape((labels.shape[0], 1))

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        # convert coordinate into polar coordinates

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cart(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple


class polar_dataset(Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False):
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        xyz = xyz.numpy().squeeze(axis=0)
        labels = labels.numpy().reshape((labels.shape[0], 1))

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 45) - np.pi / 8
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cart(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)

        return data_tuple


def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def collate_fn_BEV(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz


def collate_fn_BEV_test(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    index = [d[5] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz, index


#############################################
# Dusty-GAN synthesis dataset
class DustyGANDataset(Dataset):
    name = "DustyGAN"

    def __init__(self, root_path="data/dusty-gan",
                 splits_path="data/SemanticKITTI",
                 split="gan",
                 step=0,
                 inpaint=False,
                 labels=False,
                 return_ref=False,
                 augment=False):

        self.root_path = root_path
        self.augment = augment
        self.return_ref = return_ref
        self.labels = labels
        self.step = step
        self.inpaint = inpaint

        label_mapping = path.join(splits_path, 'semantic-kitti.yaml')
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']

        self.items = [l.strip() for l in open(path.join(splits_path, split + '_step' + str(step) + '.txt'), 'r')]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):

        fname = path.join(self.root_path, self.items[item] + '.bin')
        if self.inpaint:
            lname = path.join(self.root_path,
                              self.items[item].replace('velodyne', 'pseudo_labels/in_painting') + '.label')
        else:
            lname = path.join(self.root_path,
                              self.items[item].replace('velodyne', 'pseudo_labels/step_' + str(self.step)) + '.label')

        raw_data = np.fromfile(fname, dtype=np.float32).reshape((-1, 4))
        xyz = np.array(raw_data[:, :3])

        if self.augment:
            xyz = gan_data_preprocessing(xyz)
            raw_data = np.zeros((xyz.shape[0], 4))
            raw_data[:, :3] = xyz

        if not self.labels:
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(lname, dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        lab = np.reshape(annotated_data.astype(np.uint8), newshape=(annotated_data.shape[0],))

        data_tuple = (torch.from_numpy(xyz).unsqueeze(0), torch.from_numpy(lab))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple

