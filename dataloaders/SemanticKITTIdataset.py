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
                 setup="Sequential_masked",
                 method="CIL",
                 c2f=False,
                 return_ref=False,
                 augment=True):

        self.root_path = root_path
        self.splits_path = splits_path
        self.return_ref = return_ref

        self.split = split
        self.step = step
        self.method = method
        self.setup = setup
        self.c2f = c2f

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
            idmap = {k: v for k, v in zip(CLyaml['c2f_names']["step_"+str(self.step)].values(), self.idmap.values()) if v in np.unique(step_map)}
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

        lab = self.step_map[lab]  # map labels according to step
        mask = lab>=0
        lab = lab[mask]
        xyz = xyz[mask, :]

        data_tuple = (torch.from_numpy(xyz).unsqueeze(0), torch.from_numpy(lab))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple
