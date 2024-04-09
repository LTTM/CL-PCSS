import copy
import datetime
import os

import yaml

import LEAK

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from shutil import rmtree

import torch

from utils.SemanticKITTIdataset import SemanticKITTIDataset

torch.backends.cudnn.benchmark = True
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn

from open3d.ml.torch.models import SparseConvUnet, PointTransformer, KPFCNN, PVCNN, PointRCNN, PointPillars, RandLANet
#from model.point_transformer import PointTransformer
#from model.randlanet import RandLANet
from utils.Open3D import Open3Dataset

from utils.metrics import Metrics
from utils.losses import ClassWiseCrossEntropyLoss

seed = 12345
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


def log_pcs(writer, pts, o, y):
    p, y = o[0].detach().argmax(dim=0).cpu(), y[0].cpu()
    cy = dset.color_label(y, norm=False).reshape(-1, 3)
    cp = dset.color_label(p, norm=False).reshape(-1, 3)
    my = y.flatten() > 0

    if my.float().sum() > 0:
        writer.add_mesh("labels", vertices=pts[:, my], colors=np.expand_dims(cy[my], 0), global_step=e)
        writer.add_mesh("preds", vertices=pts[:, my], colors=np.expand_dims(cp[my], 0), global_step=e)


def conv_to_tensor(inputs):
    data = {}
    for k, v in inputs.items():
        if k == "coords" or k == "neighbor_indices" or k == "sub_idx" or k == "interp_idx":
            new = [torch.from_numpy(t) for t in v]
            data = {**data, **{k: new}}
        else:
            data = {**data, **{k: torch.from_numpy(v)}}
    return data


def validate(writer, vset, vloader, epoch, model, device):  # PA, PP, mIoU
    metric = Metrics(vset.pointcloud_dataset.cnames, device=device, mask=True)
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(vloader, "Validating Epoch %d" % (epoch + 1), total=int(len(vset)/vloader.batch_size))):
            o = model(data)
            y = data["labels"].cuda()
            metric.add_sample(o.argmax(dim=2).flatten(), y.flatten())

    miou = metric.percent_mIoU()
    acc = metric.percent_acc()
    prec = metric.percent_prec()
    writer.add_scalar('mIoU', miou, epoch)
    writer.add_scalar('PP', prec, epoch)
    writer.add_scalar('PA', acc, epoch)
    writer.add_scalars('IoU', {n: 100 * v for n, v in zip(metric.name_classes, metric.IoU()) if not torch.isnan(v)},
                       epoch)
    print(metric)
    model.train()
    return miou, o.swapaxes(1, 2), y


if __name__ == '__main__':
    epochs = 20 #50
    batch_size = 1 #6
    val_batch =  1 #20
    CL = False
    c2f = False
    step = 0
    setup = "Sequential_Masked"
    cube_edge = 96
    val_cube_edge = 96
    num_classes = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Open3Dataset

    if CL:
        pretrain = "step0"
        #pretrain = "step"+str(step-1) #"step0__18-54-36" #pcs_semantic-kitti_randlanet_bis"  # "s3dis"
        logdir = "log/train" + "_step" + str(step) + "_NEW_" + setup + "_" + datetime.datetime.now().strftime("%H-%M-%S") + pretrain
    else:
        pretrain = "" #kitti_randlanet_47"
        logdir = "log/train_kitti_pt_" + datetime.datetime.now().strftime("%H-%M-%S") + pretrain
    rmtree(logdir, ignore_errors=True)
    writer = SummaryWriter(logdir, flush_secs=.5)

    my_model = PointTransformer

    model = my_model(device='cuda', num_classes=num_classes)  # SimpleNet(num_classes)

    if pretrain:
        new = model.state_dict()
        old = torch.load("log/train_" + pretrain + "/val_best.pth")
        for k in new:
            if "fc1.3" not in k: #out
                new[k] = old[k]
            else:
                if new[k].shape == old[k].shape:
                    new[k] = old[k]
        model.load_state_dict(new)

    model.to('cuda')

    dset = dataset(pointcloud_dataset=SemanticKITTIDataset(split='train',
                                                           ues=False,
                                                           CL=CL,
                                                           c2f=c2f,
                                                           step=step,
                                                           setup=setup),
                                                           model = my_model)
        #root_path="../PCSproject/Nuvole_di_punti", fsl=15,
                   #cube_edge=cube_edge)
    dloader = DataLoader(dset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4,
                         drop_last=True)

    vset = dataset(pointcloud_dataset=SemanticKITTIDataset(augment=False, split='val',
                                                           CL=CL,
                                                           c2f=True,
                                                           step=step,
                                                           setup=setup),
        #root_path="../PCSproject/Nuvole_di_punti",
                   #cube_edge=val_cube_edge,
                   augment=False,
                   split='val')
    vloader = DataLoader(vset,
                         batch_size=val_batch,
                         shuffle=False,
                         num_workers=4)

    steps_per_epoch = len(dset) // batch_size
    tot_steps = steps_per_epoch * epochs
    lr0 = 1e-2 #2.5e-4 #1e-2
    #lre = 1e-5 #1e-5
    lr_decays = {i: 0.95 for i in range(0, 500)}

    optim = Adam(model.parameters()) #, weight_decay=1e-5)

    pts = 2 * torch.from_numpy(np.indices((val_cube_edge, val_cube_edge, val_cube_edge)).reshape(3, -1).T).unsqueeze(
        0) / cube_edge - 1.

    best_miou = 0

    for e in range(epochs):
        torch.cuda.empty_cache()
        if e % 1 == 0:
            if e > 0:
                miou, o, y = validate(writer, vset, vloader, e, model, device)
                if miou > best_miou:
                    best_miou = miou
                    torch.save(model.state_dict(), logdir + "/val_best.pth")
                # log_pcs(writer, pts, o, y)
            metrics = Metrics(vset.pointcloud_dataset.cnames, device=device, mask=True)
            #metrics = Metrics([n for i,n in enumerate(vset.pointcloud_dataset.cnames) if i in [0,1,6,9,13,15,18]], device=device, mask=True)

        pbar = tqdm(dloader, total=steps_per_epoch,
                    desc="Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress" % (e + 1, epochs, 0., 0.))

        for i, data in enumerate(pbar):
            step = i + steps_per_epoch * e

            #lam = schedule(10, 1, step, tot_steps // 2, .9)
            data_config = os.path.join('utils', 'LEAK_Utils.yaml')
            DATA = yaml.safe_load(open(data_config, 'r'))
            loss = nn.CrossEntropyLoss(ignore_index=-1)
            fair_loss = LEAK.FairLoss(alpha=10., mapping=DATA["micro2macro_SemanticKITTI"],
                                      idx=DATA["macro_idx_SemanticKITTI"], classes=num_classes)
            proto_loss = LEAK.ProtoLoss(ignore_index=-1, gamma=1e-3, classes=num_classes)
            macro_proto_loss = LEAK.ProtoLoss(ignore_index=-1, gamma=1e-5, classes=num_classes)
            prototypes = torch.zeros((num_classes, 512)).cuda()
            macro_prototypes = torch.zeros((len(DATA["macro_idx_SemanticKITTI"]), 512)).cuda()
            K = torch.zeros((num_classes)).cuda()
            M = torch.zeros((len(DATA["macro_idx_SemanticKITTI"]))).cuda()

            lr = lr0 #schedule(lr0, lre, step, tot_steps, .95)
            optim.param_groups[0]['lr'] = lr
            lr = adjust_learning_rate(optim, lr_decays, e)

            optim.zero_grad()

            # x, y = x.to(device), y.to(device, dtype=torch.long)-1 # shift indices
            o = model(data)
            y = data["label"].cuda()

            #l,_,_ = model.get_loss(loss, o, data, device) # non è ancora weighted
            l = loss(o.swapaxes(1, 2), y.long())  + fair_loss(o.swapaxes(1, 2), y.long()) #
            l.backward()

            metrics.add_sample(o.detach().argmax(dim=2).flatten(), y.flatten())

            optim.step()
            miou = metrics.percent_mIoU()
            pbar.set_description("Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress" % (e + 1, epochs, l.item(), miou))

            writer.add_scalar('lr', lr, step)
            writer.add_scalar('loss', l.item(), step)
            writer.add_scalar('step_mIoU', miou, step)

        torch.save(model.state_dict(), logdir + "/latest.pth")

    miou, o, y = validate(writer, vset, vloader, e, model, device)
    if miou > best_miou:
        best_miou = miou
        torch.save(model.state_dict(), logdir + "/val_best.pth")
