" UTILITY FUNCTIONS"

import torch
import numpy as np

def set_seed(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def adjust_learning_rate(optimizer, lr_decays, epoch):
    lr = optimizer.param_groups[0]['lr']
    lr = lr * lr_decays[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def schedule(lr0, lre, step, steps, power):
    return (lr0 - lre) * (1 - min(step / steps, 1)) ** power + lre  # learning rate decrease poly 0.9

def log_pcs(writer, dset, e, pts, o, y):
    p, y = o[0].detach().argmax(dim=0).cpu(), y[0].cpu()
    cy = dset.color_label(y, norm=False).reshape(-1, 3)
    cp = dset.color_label(p, norm=False).reshape(-1, 3)
    my = y.flatten() > 0

    if my.float().sum() > 0:
        writer.add_mesh("labels", vertices=pts[:, my], colors=np.expand_dims(cy[my], 0), global_step=e)
        writer.add_mesh("preds", vertices=pts[:, my], colors=np.expand_dims(cp[my], 0), global_step=e)