import os

# Set environment variable for CUDA launch blocking
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import yaml
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from shutil import rmtree

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn
torch.backends.cudnn.benchmark = True

from model.randlanet import RandLANet
from utils.Open3D import Open3Dataset
from utils.metrics import Metrics
from utils.argparser import init_params
from utils.utils import adjust_learning_rate

# Define step classes for continual learning
step_classes = {0: [1, 2, 3],
                1: [1, 2, 3, 4, 5, 6],
                2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}

##### Validation Routine
def validate(writer, vset, vloader, epoch, model, device, args):

    # Initialize metrics
    metric = Metrics(vset.pointcloud_dataset.cnames, device=device, mask_unlabeled=True, mask=list(np.arange(len(step_classes[args.CLstep])+1)))
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(vloader, "Validating Epoch %d" % (epoch + 1), total=int(len(vset) / vloader.batch_size))):
            o, _ = model(data)
            y = data["labels"].cuda()
            metric.add_sample(o.argmax(dim=2).flatten(), y.flatten())

    # Calculate metrics
    miou = metric.percent_mIoU()
    acc = metric.percent_acc()
    prec = metric.percent_prec()
    writer.add_scalar('mIoU', miou, epoch)
    writer.add_scalar('PP', prec, epoch)
    writer.add_scalar('PA', acc, epoch)
    writer.add_scalars('IoU', {n: 100 * v for n, v in zip(metric.name_classes, metric.IoU()) if not torch.isnan(v)}, epoch)
    print(metric)
    model.train()
    return miou, o.swapaxes(1, 2), y


def main(args):
    # Set device to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = args.dataset
    args.c2f = True

    # Print configuration
    print(f"Training SemanticKITTI using RandLA-Net step {args.CLstep}")
    print("C2F configuration")

    # Get datasets with continual learning configurations
    dset = Open3Dataset(pointcloud_dataset=dataset(split='train',
                                                   CL=True,
                                                   c2f=args.c2f,
                                                   step=args.CLstep))
    dloader = DataLoader(dset,
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=args.num_workers,
                         drop_last=True)
    print(dset.pointcloud_dataset.idmap)
    print("*****************************************")

    vset = Open3Dataset(pointcloud_dataset=dataset(augment=False, split='val',
                                                   CL=True,
                                                   c2f=args.c2f,
                                                   step=args.CLstep))
    vloader = DataLoader(vset,
                         batch_size=args.val_batch_size,
                         shuffle=False,
                         num_workers=args.num_workers)

    # Initialize model
    model = RandLANet(num_neighbors=16, device='cuda', num_classes=len(step_classes[args.CLstep])+1)

    # Set up logging directory
    logdir = os.path.join(args.logdir, "train_" + args.test_name)
    rmtree(logdir, ignore_errors=True)
    writer = SummaryWriter(logdir, flush_secs=.5)
    
    # Load pretrained model if specified
    if args.pretrained_model:
        new = model.state_dict()
        old = torch.load(args.ckpt_file)

        # Load weights but exclude the last layer
        if args.CL and args.CLstep > 0:
            for k in new:
                if "fc1.3" not in k:  # out
                    new[k] = old[k]
                else:
                    if new[k].shape == old[k].shape:
                        new[k] = old[k]
        model.load_state_dict(new)
    model.to('cuda')
    
    # Training parameters
    steps_per_epoch = len(dset) // args.batch_size
    lr0 = args.lr  # Initial learning rate
    lr_decays = {i: args.poly_power for i in range(0, args.decay_over_iterations)}
    optim = Adam(model.parameters(), weight_decay=args.weight_decay)

    best_miou = 0

    for e in range(args.epochs):
        torch.cuda.empty_cache()

        # Validation routine
        if e % args.eval_every_n_epochs == 0 and e > 0:
            miou, o, y = validate(writer, vset, vloader, e, model, device, args)
            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), logdir + "/val_best.pth")

        metrics = Metrics(vset.pointcloud_dataset.cnames, device=device, mask_unlabeled=True)

        pbar = tqdm(dloader, total=steps_per_epoch,
                    desc="Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress" % (e + 1, args.epochs, 0., 0.))

        # Training routine
        for i, data in enumerate(pbar):
            step = i + steps_per_epoch * e

            loss = nn.CrossEntropyLoss(ignore_index=0)

            lr = lr0  # Initial learning rate
            optim.param_groups[0]['lr'] = lr
            lr = adjust_learning_rate(optim, lr_decays, e)

            optim.zero_grad()

            o, _ = model(data)
            
            ############################## C2F LOSS #################################
            CL_mapping = os.path.join(dset.pointcloud_dataset.splits_path, dset.pointcloud_dataset.method, 'CILconfig.yaml')  # CIL
            with open(CL_mapping, 'r') as stream:
                CLyaml = yaml.safe_load(stream)
            c2f_step_map = CLyaml['c2f']['step_' + str(args.CLstep)]
            step_map = np.vectorize(c2f_step_map.__getitem__)(np.arange(0, len(c2f_step_map)))
            idmap = {k: v for k, v in dset.pointcloud_dataset.idmap.items() if v in np.unique(step_map)}
            y = torch.from_numpy(step_map.__getitem__(data["labels"])).cuda()            

            l = loss(o.swapaxes(1, 2).contiguous(), y)
            l.backward()

            pred = o.argmax(dim=2).flatten()
            metrics.add_sample(pred.flatten(), y.flatten())

            optim.step()
            miou = metrics.percent_mIoU()
            pbar.set_description("Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress" % (e + 1, args.epochs, l.item(), miou))

            writer.add_scalar('lr', lr, step)
            writer.add_scalar('loss', l.item(), step)
            writer.add_scalar('step_mIoU', miou, step)

        torch.save(model.state_dict(), logdir + "/latest.pth")

    # Final validation
    miou, o, y = validate(writer, vset, vloader, e, model, device, args)
    if miou > best_miou:
        best_miou = miou
        torch.save(model.state_dict(), logdir + "/val_best.pth")


if __name__ == '__main__':
    args = init_params('train_c2f', verbose=True)
    main(args)