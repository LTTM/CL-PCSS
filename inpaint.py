from os.path import exists, join

import numpy as np
import os
import random
import torch
import yaml
from torch import nn
from tqdm import tqdm

import wandb
torch.backends.cudnn.benchmark = True
from torch.optim import Adam

from utils.lovasz_losses import lovasz_softmax
from utils.argparser import init_params, print_cfg
from utils.metric_util import fast_hist_crop, fast_hist

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def schedule(lr0, lre, step, steps, power):
    return (lr0 - lre) * (1 - min(step / steps, 1)) ** power + lre  # learning rate decrease poly 0.9


#########################################################################
#### utils functions

# function to inpaint labels on a [2,n] array
# where the second row correspond to actual labels, while first row to inpainted labels
def inpaint(m):
    labs = np.zeros((m.shape[1]))
    # questo è l'algoritmo per 2 sole
    # TODO: implementare versione con più labels
    for i in range(m.shape[1]):
        if m[1,i]==0:
            labs[i] = m[0,i]
        else:
            labs[i] = m[1,i] #se sono diverse assegno la classe data dall'ultimo modello
    return labs.astype(np.uint32)

# faccio predizione dal modello vecchio su dset nuovo e su tutte quelle che son zero ci metto il valore predetto, altrimenti la label
def train_inpainting(dloader, model, batch, cnames, t1=0.1, t2=0.7):
    model.eval()
    with torch.no_grad():
        out_labels_path = 'checkpoints/CIL/sequences/'
        data_config = 'config/label_mapping/semantic-kitti.yaml'
        DATA = yaml.safe_load(open(data_config, 'r'))
        remap_dict = DATA["learning_map_inv"]
        # make lookup table for mapping
        max_key = max(remap_dict.keys())
        remap_lut = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
        remap_dict_val = DATA["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())
        gbar = tqdm(dloader, desc="Train inpainting")
        for i, (xyz, vox_label, grid, pt_labs, pt_fea) in enumerate(gbar):
            #inpaint only a percentage of training set
            #sampling = np.round(1/percentage)
            #TODO: inpaint only a percentage of the training set
            file_name = dloader.dataset.point_cloud_dataset.im_idx[i]
            frame = file_name.split('/')[-1]
            scan_number = file_name.split('/')[-3]
            store_path = join(out_labels_path, scan_number, 'predictions')
            os.makedirs(store_path) if not exists(store_path) else None
            store_file = join(store_path, str(frame[:-4]) + '.label')
            multi_labs = []
            # predict labels
            hist_list = []
            (xyz, vox_label, grid, pt_labs, pt_fea) = dloader.dataset[i]
            grid = [grid]
            pt_fea = [pt_fea]
            pt_labs = [pt_labs]
            pt_fea_ten = [torch.from_numpy(j).type(torch.FloatTensor).to('cuda') for j in pt_fea]
            grid_ten = [torch.from_numpy(j).to('cuda') for j in grid]
            o, feats = model(pt_fea_ten, grid_ten, batch)
            predict_labels = torch.nn.functional.softmax(o, dim=1) #qui
            predict_idx = torch.argmax(predict_labels, dim=1) #qui
            predict_labels = predict_idx.cpu().detach().numpy()
            predict_labels = predict_labels.astype(np.uint8)
            for count, i_grid in enumerate(grid):
                hist_list.append(fast_hist_crop(predict_labels[
                                      count, grid[count][:, 0], grid[count][:, 1],
                                      grid[count][:, 2]], pt_labs[count], cnames))
            gc = len(grid) - 1
            labs = predict_labels[gc, grid[gc][:, 0], grid[gc][:, 1], grid[gc][:, 2]]
            #for x in range(batch):
            # constrain to have softmax pred over a certain thresh
            predict_values = torch.nn.functional.softmax(o, dim=1)
            predict_values0 = torch.max(predict_values, dim=1)
            predict_values = predict_values0[0].detach().cpu().numpy()
            vals = predict_values[gc, grid[gc][:, 0], grid[gc][:, 1], grid[gc][:, 2]]
            remove = vals < t2 # * np.max(vals)
            labs[remove] = 0
            #constrain to have difference b/t first and second max over a threshold
            #predict_idx = torch.argmax(predict_labels, dim=1)
            predict_values = torch.nn.functional.softmax(o, dim=1)
            predict_values, _ = torch.topk(predict_values, 2, dim=1)
            predict_values1 = predict_values[:,0].detach().cpu().numpy()
            predict_values2 = predict_values[:,1].detach().cpu().numpy()
            pv1 = predict_values1[gc, grid[gc][:, 0], grid[gc][:, 1], grid[gc][:, 2]]
            pv2 = predict_values2[gc, grid[gc][:, 0], grid[gc][:, 1], grid[gc][:, 2]]
            delta = pv1 - pv2
            remove = delta < t1 # * np.max(vals)
            labs[remove] = 0
            # constrain to have only previous step labels
            labbs = np.zeros(labs.shape, dtype=np.uint32)
            #mapp = {k: v for k, v in DATA["labels"].items()}
            for l in cnames:
                mask = (labs == remap_lut_val[l])
                labbs[mask] = remap_lut_val[l]
            multi_labs.append(labbs.reshape(-1, 1))
            lab = np.array(pt_labs, dtype=np.uint32).reshape(-1, 1)
            multi_labs.append(lab)
            # effective inpainting
            multi_labs = np.squeeze(np.array(multi_labs))
            if multi_labs.shape[0] == 1:
                labs = multi_labs[0]
            else:
                labs = inpaint(multi_labs)
            #debug point clouds
            #log_pcs(dset.point_cloud_dataset[0][0], labbs.reshape(-1), dset.point_cloud_dataset.cmap, name='refined-prediction')
            #log_pcs(dset.point_cloud_dataset[0][0], lab.reshape(-1), dset.point_cloud_dataset.cmap, name='inpaint')
            #log_pcs(dset.point_cloud_dataset[0][0], dset.point_cloud_dataset[0][1], dset.point_cloud_dataset.cmap,
            #        name='original')
            #log_pcs(dset.point_cloud_dataset[0][0], labs.reshape(-1), dset.point_cloud_dataset.cmap, name='inpainted')
            # produce voting labels
            labs = labs.astype(np.uint32)
            #upper_half = labs >> 16  # get upper half for instances
            lower_half = labs & 0xFFFF  # get lower half for semantics
            labs = remap_lut[lower_half]  # do the remapping of semantics
            #labs = (upper_half << 16) + lower_half  # reconstruct full label
            labs = labs.astype(np.uint32)
            os.makedirs(store_path) if not exists(store_path) else None
            labs.tofile(store_file)
    model.train()
