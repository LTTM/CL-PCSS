from os import path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from open3d._ml3d.datasets.base_dataset import BaseDataset
from torch.utils.data import Dataset
from collections import namedtuple

from model.randlanet import RandLANet
from open3d.ml.torch.models import SparseConvUnet, PointTransformer, KPFCNN, PVCNN, PointRCNN, PointPillars, RandLANet

class Open3Dataset(Dataset):
    def __init__(self,
                 name="DICEA",
                pointcloud_dataset = None,
                root_path="../semantic_kitti",
                splits_path="data/SemanticKITTI",
                split = "train",
                num_pts = 122880,
                augment = True,
                repeat = 1,
                model = PVCNN,
                **kwargs):
                super().__init__() #name=name)

                self.pointcloud_dataset = pointcloud_dataset
                self.split = split
                self.root_path = root_path
                self.num_pts = num_pts
                self.augment = augment

                self.model = model

                self.items = pointcloud_dataset.items
                #self.items = repeat * [l.strip() for l in open(path.join(splits_path, split + '.txt'), 'r')]
                self.path_list = self.items

    def __getitem__(self, item):
        model = self.model(num_classes=len(self.pointcloud_dataset.idmap))
        data = self.getitem(item)
        inputs = {"point": data[0].squeeze(0).float(), "feat": data[0].squeeze(0).float(), "label": data[1]}
        inputs = model.preprocess(inputs, self.get_attr(item))
        inputs = model.transform(inputs, self.get_attr(item))
        #inputs = namedtuple('Struct',inputs.keys())(*inputs.values())
        #del inputs["search_tree"]
        return inputs

    def __len__(self):
        return len(self.items)

    def color_label(self, lab, norm=True):
        if norm:
            return self.pointcloud_dataset.cmap[lab.numpy()] / 255.
        else:
            return self.pointcloud_dataset.cmap[lab.numpy()]

    def getitem(self, item):
        return self.pointcloud_dataset[item]

    def to_plottable(self, x):
        return x.transpose(0, 2)

    ###################################
    # open3d dataset functions
    def get_split(self, split):
        return self.split

    def is_tested(self, attr):
        pass
        # checks whether attr['name'] is already tested.

    def save_test_result(self, results, attr):
        pass
        # save results['predict_labels'] to file.

    def get_data(self, idx):
        data = self.getitem(idx)
        points, features, labels = data[0], None, data[1]
        return {'point': points, 'feat': features, 'label': labels}

    def get_attr(self, idx):
        path = self.path_list[idx]
        name = path.split('/')[-1]
        return {'name': name, 'path': path, 'split': self.split}
