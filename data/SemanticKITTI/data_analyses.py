from os import listdir, path
import numpy as np
import torch, yaml

sfolder = "../../../RandLA-Net/data/semantic_kitti/dataset/sequences"
splits_path=""
trainlist = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]

label_mapping = path.join(splits_path, 'semantic-kitti.yaml')
with open(label_mapping, 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
learning_map = semkittiyaml['learning_map']

ss = open("seq_statistics.txt", "w")
pc = open("pcs_statistics.txt", "w")

########
# (1) Count the number of point clouds per each sequence
for t in trainlist:
    root_path = path.join(sfolder, t, "velodyne")
    items = listdir(root_path)
    #print("Samples in sequence\t %s: %d" % (t, len(items)))
    ss.write("Samples in sequence\t %s: %d" % (t, len(items)))
    ss.write("\n")
ss.close()

########
# (2) Count the number of points per each point cloud per each sequence subdivided by class
for t in trainlist:
    root_path = path.join(sfolder, t, "velodyne")
    items = listdir(root_path)

    for item in items:
        item = item[:-4]
        fname = path.join(root_path, item + '.bin')
        lname = path.join(root_path.replace('velodyne', 'labels'), item + '.label')

        raw_data = np.fromfile(fname, dtype=np.float32).reshape((-1, 4))

        annotated_data = np.fromfile(lname, dtype=np.uint32).reshape((-1, 1))
        annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
        annotated_data = np.vectorize(learning_map.__getitem__)(annotated_data)

        xyz = np.array(raw_data[:, :3])
        lab = np.reshape(annotated_data.astype(np.uint8), newshape=(annotated_data.shape[0],))
        u,c = np.unique(lab, return_counts=True)
        pts = np.zeros(20)
        pts[u] = c
        #print("Points in sample\t %s\t in sequence\t %s: " % (item, t), pts)
        pc.write("Seq %s, sample %s: " % (t, item))
        for p in pts:
            pc.write("%ld "%(p))
        pc.write("\n")
pc.close()