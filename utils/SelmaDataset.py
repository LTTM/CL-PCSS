import numpy as np
import torch
from plyfile import PlyData

from numpy.lib.recfunctions import structured_to_unstructured
from scipy.spatial.transform import Rotation as R

import sys, os

from utils.Open3D import Open3Dataset

sys.path.append(os.path.abspath('.'))

class SELMADataset(Open3Dataset):
    def __init__(self,
                 split_extension='txt',
                 split_separator='\n',
                 split_skiplines=0,
                 image_label=False,
                 **kwargs): # whether to use city19 or city36 class set

        super(SELMADataset, self).__init__(split_extension=split_extension,
                                          split_separator=split_separator,
                                          split_skiplines=split_skiplines,
                                          **kwargs)
        
        self.image_label = image_label

        self.cpath = ["right", "back", "left", "front"]
        self.lipath = "lidar"

        self.ltrans = np.array([-.65, 0, 1.7]) # lidar translation
        self.lrot = R.from_euler('xyz', [0, 0, -90], degrees=True).as_matrix() # lidar rotation -- roll, pitch, yaw
        self.ilrot = np.linalg.inv(self.lrot)

        self.ctrans = {'front': np.array([.2, 0, 1.5]), # front camera translation
                       'right': np.array([-.65, .45, 1.5]), # right camera translation
                       'back': np.array([-1.5, 0, 1.5]), # back camera translation
                       'left': np.array([-.65, -.45, 1.5])} # left camera translation

        self.crot = {'front': R.from_euler('xyz', [0, 5, 0], degrees=True).as_matrix(), # front camera rotation (angles are inverted w.r.t. the sensor rotation)
                     'right': R.from_euler('xyz', [0, 0, -90], degrees=True).as_matrix(), # right camera rotation (angles are inverted w.r.t. the sensor rotation)
                     'back': R.from_euler('xyz', [0, 0, -180], degrees=True).as_matrix(), # back camera rotation (angles are inverted w.r.t. the sensor rotation)
                     'left': R.from_euler('xyz', [0, 0, -270], degrees=True).as_matrix()} # left camera rotation (angles are inverted w.r.t. the sensor rotation)
        self.icrot = {k: np.linalg.inv(self.crot[k]) for k in self.crot}

        self.K = np.array([[640, 640, 0], [320, 0, -640], [1, 0, 0]]) # camera parameters (intrinsics)
        self.iK = np.linalg.inv(self.K)

        self.H, self.W = self.K[1,0]*2, self.K[0,0]*2
        self.crop = 140 # minimum crop, fixed for batch-size consistency
        
        self.max_depth = 100
        self.depth = self.max_depth*np.ones((self.H-self.crop, self.W), np.float32)
        self.label = self.ignore_index*np.ones((self.H-self.crop, self.W), np.int64)

        uv = np.meshgrid(np.arange(self.W), np.arange(self.crop, self.H), [1])
        self.uv = np.stack(uv).reshape(3,-1).T.astype(np.float32)

        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * 3 + 1, 2 * 7 + 1), (3,7)) # initialize the elliptical kernel (taller)

    def init_ids(self):
        self.raw_to_train = {-1:-1, 0:-1, 1:2, 2:4, 3:-1, 4:-1, 5:5, 6:0, 7:0, 8:1, 9:8, 10:-1,
                             11:3, 12:7, 13:10, 14:-1, 15:-1, 16:-1, 17:-1, 18:6, 19:-1, 20:-1,
                             21:-1, 22:9, 40:11, 41:12, 100:13, 101:14, 102:15, 103:16, 104:17,
                             105:18, 255:-1}
        self.ignore_index = -1

    # function to project the input pointcloud (lidar coords) to camera PoV
    # pc is a list of xyz points
    # ltrans is a vector (for the lidar)
    # lrot is a rotation matrix (for the lidar)
    # ctrans is a vector (for the camera)
    # crot is a rotation matrix (for the camera)
    def project(self, pc, view):
        pc = pc@self.lrot.T                          # lidar rotation as matrix
        pc += self.ltrans                            # lidar translation as vector
        pc -= self.ctrans[view]                      # camera translation as vector
        pc = pc@self.crot[view].T                    # camera rotation as matrix (inverse)

        p = pc @ self.K.T                            # project world coords in camera uv (@: matrix multiplication)
        
        d = p[:, -1:].copy()                         # depth
        ids = np.argsort(d[:,0])[::-1]               # sort by depth, closer points override far points
        p = p[ids]                                   # reorder pc
        d = d[ids]                                   # reorder depth
        
        m = d[:, 0] > 0                              # get mask for valid depths (i.e. not behind the camera plane)
        p /= np.clip(d, a_min=1e-15, a_max=np.inf)   # clip depths 

        m = np.logical_and(m, p[:, 0] >= 0)          # mask valid projected points 
        m = np.logical_and(m, p[:, 0] < self.W)      # mask valid projected points 
        m = np.logical_and(m, p[:, 1] >= 0)          # mask valid projected points 
        m = np.logical_and(m, p[:, 1] < self.H)      # mask valid projected points
        
        p = np.floor(p[m]).astype(int)               # cast to int the indices
        return p, d[m,0], m, ids

    def unproject(self, depth, view):
        m = self.uv*depth.reshape(-1,1)              # scale the pixel coordinates by the depth
        pc = m @ self.iK.T                           # unproject using the camera parameters
        pc = pc @ self.icrot[view].T                 # rotate (camera)
        pc += self.ctrans[view]                      # shift (camera)
        pc -= self.ltrans                            # shift (lidar)
        pc = pc @ self.ilrot.T                       # rotate (lidar)
        return pc.reshape(self.H-self.crop, self.W, 3)

    def to_range(self, tensor, wandb=False):
        r = np.array(tensor[0:1].transpose(0,1).transpose(1,2) + 100.)
        r = r/200.
        if wandb:
            return np.round(255*r).astype(np.uint8)
        return r
    
    def to_rgb(self, tensor, wandb=False):
        t = np.array(tensor.transpose(0,1).transpose(1,2))+127.5
        t = t/255.
        if wandb:
            return np.round(255*t).astype(np.uint8)
        return t
    
    def to_xyz(self, tensor, wandb=False):
        tensor = tensor[1:]
        t = np.array(tensor.transpose(0,1).transpose(1,2))+100.
        t = t / 200.
        if wandb:
            return np.round(255*t).astype(np.uint8)
        return t

    def __getitem__(self, item):
        (fname,) = self.items[item]
        
        lidar = PlyData.read(os.path.join(self.root_path, self.lipath, fname+".ply"))
        pc = structured_to_unstructured(lidar["vertex"][['x', 'y', 'z']])     # get xyz
        pcl_raw = structured_to_unstructured(lidar["vertex"][['ObjTag']])     # get labels

        data_tuple = (torch.from_numpy(pc).unsqueeze(0), torch.from_numpy(pcl_raw))
        return data_tuple