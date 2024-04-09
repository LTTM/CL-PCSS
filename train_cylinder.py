import numpy as np
import wandb

from tqdm import tqdm
from tensorboardX import SummaryWriter
from shutil import rmtree
from torch import nn
import torch, random, os

from utils.SemanticKITTIdataset import SemanticKITTIDataset, cylinder_dataset, voxel_dataset, polar_dataset, collate_fn_BEV, collate_fn_BEV_test
from utils.lovasz_losses import lovasz_softmax

torch.backends.cudnn.benchmark = True
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.segcloud import ESegCloud
from model.cylinder3d.cylinder3d import Cylinder3D

from utils.metrics import Metrics

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def schedule(lr0, lre, step, steps, power):
    return (lr0 - lre) * (1 - min(step / steps, 1)) ** power + lre  # learning rate decrease poly 0.9


def log_pcs(writer, dset, o, y):
    my_pred = o.detach().cpu().numpy()
    my_pred = np.argmax(my_pred, axis=1).flatten()
    my_gt = y.detach().cpu().numpy().flatten()
    my_pc = dset[0][0].reshape(3, my_gt.shape[0])
    cy = dset.point_cloud_dataset.color_label(my_gt, norm=False).reshape(-1, 3)[:, :3]
    cp = dset.point_cloud_dataset.color_label(my_pred, norm=False).reshape(-1, 3)[:, :3]
    my = y.flatten() > 0
    gt = np.vstack([my_pc, my_gt.reshape(1, len(my_gt))])
    p = np.vstack([my_pc, my_pred.reshape(1, len(my_pred))])

    # cosi logga i voxels
    wandb.log({"point_cloud_gt": wandb.Object3D(gt.T)})
    wandb.log({"point_cloud_pred": wandb.Object3D(p.T)})

    # if my.float().sum()>0:
    #    writer.add_mesh("labels", vertices=my_pc[:, my], colors=np.expand_dims(cy[my], 0), global_step=e)
    #    writer.add_mesh("preds", vertices=my_pc[:, my], colors=np.expand_dims(cp[my], 0), global_step=e)


def my_collate(batch):
    vox = [item[0] for item in batch]
    vox_lab = [item[1] for item in batch]
    data = [item[2] for item in batch]
    data_lab = [item[3] for item in batch]
    target = [item[4] for item in batch]
    return [vox, vox_lab, data, data_lab, target]


def validate(writer, vset, vloader, epoch, model, device):  # PA, PP, mIoU
    metric = Metrics(vset.point_cloud_dataset.cnames, device=device)
    model.eval()
    with torch.no_grad():
        vbar = tqdm(vloader, "Validating Epoch %d" % (epoch + 1), total=len(vset))
        for i, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(vbar):
            val_pt_fea_ten = [torch.from_numpy(j.detach().numpy()).type(torch.FloatTensor).to(device) for j in
                              val_pt_fea]
            val_grid_ten = [torch.from_numpy(j.detach().numpy()).to(device) for j in val_grid]
            y = val_vox_label.type(torch.LongTensor).to(device)
            o = model(val_pt_fea_ten, val_grid_ten)
            metric.add_sample(o.argmax(dim=1).flatten(), y.flatten())
    miou = metric.percent_mIoU()
    wandb.log({"valid_miou": miou})
    writer.add_scalar('mIoU', miou, epoch)
    print(metric)
    model.train()
    return miou, o, y


class Trainer():
    def __init__(self, args):

        wandb.login(key="8dda0c466d22875b3c20a7f92b1218c0afb3ae66")
        wandb.init(project="xxx", entity="elena_camuffo", mode="disabled")
        # wandb.config = {
        #    "learning_rate": args.lr,
        #    "epochs": args.epochs,
        #    "batch_size": args.batch_size
        # }

        # load_model = args.pretrained_model

        eval_every_n_epochs = 10
        epochs = 25
        batch_size = 1 # 8
        val_batch_size = 1 #1
        cube_edge = 96
        val_cube_edge = 96
        num_classes = 20 # 8
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = SemanticKITTIDataset
        arch = Cylinder3D

        test_name = arch.name + "_PCS_fsl-all"
        logdir = os.path.join("log/train", test_name)
        # rmtree(logdir, ignore_errors=True) #rimuove i files di log vecchi
        writer = SummaryWriter(logdir, flush_secs=.5)

        model = arch(num_classes=num_classes)

        model.to(device)
        dset = cylinder_dataset(dataset(), #root_path="../PCSproject/Nuvole_di_punti",
                             grid_size=model.output_shape)
        dloader = DataLoader(dset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2,
                             collate_fn=my_collate,
                             drop_last=True)

        vset = voxel_dataset(dataset( #root_path="../PCSproject/Nuvole_di_punti", cube_edge=val_cube_edge,
                                     augment=False,
                                     split='val'), grid_size=model.output_shape)
        vloader = DataLoader(vset,
                             batch_size=val_batch_size,
                             # collate_fn=my_collate,
                             shuffle=False,
                             num_workers=2)

        loss = nn.CrossEntropyLoss(ignore_index=-1)

        steps_per_epoch = len(dset) // batch_size
        tot_steps = steps_per_epoch * epochs
        lr0 = 1e-3 #2.5e-4
        lre = 1e-5
        optim = Adam(model.parameters(), lr=lr0, weight_decay=1e-2) #5)
        best_miou = 0

        wandb.watch(model)

        ##########################################
        # TRAINING
        for e in range(epochs):
            torch.cuda.empty_cache()
            if e % eval_every_n_epochs == 0:
                if e >= 0:
                    miou, o, y = validate(writer, vset, vloader, e, model, device)
                    if miou > best_miou:
                        best_miou = miou
                        torch.save(model.state_dict(), logdir + "/val_best.pth")
                    # log_pcs(writer, dset, o, y)
                metrics = Metrics(dset.point_cloud_dataset.cnames, device=device)

            pbar = tqdm(dloader, total=steps_per_epoch,
                        desc="Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress" % (e + 1, epochs, 0., 0.))

            for i, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(pbar):
                step = i + steps_per_epoch * e
                lr = schedule(lr0, lre, step, tot_steps, .9)
                optim.param_groups[0]['lr'] = lr
                optim.zero_grad()

                train_pt_fea_ten = [torch.from_numpy(j).type(torch.FloatTensor).to(device) for j in train_pt_fea]
                train_grid_ten = [torch.from_numpy(j).to(device) for j in train_grid]
                y = torch.from_numpy(np.array(train_vox_label)).type(torch.LongTensor).to(device)

                o = model(train_pt_fea_ten, train_grid_ten)

                l = loss(o, y) # + lovasz_softmax(torch.argmax(o), y)
                l.backward()

                p = o.detach().argmax(dim=1)
                metrics.add_sample(p.flatten(), y.flatten())

                optim.step()
                miou = metrics.percent_mIoU()
                pbar.set_description("Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress" % (e + 1, epochs, l.item(), miou))

                writer.add_scalar('lr', lr, step)
                writer.add_scalar('loss', l.item(), step)
                writer.add_scalar('step_mIoU', miou, step)
                writer.add_scalars('IoU', {n: 100 * v for n, v in zip(metrics.name_classes, metrics.IoU())}, step)

                wandb.log({"learning_rate": lr})
                wandb.log({"train_loss": l.item()})
                wandb.log({"step_mIoU": miou})
                # wandb.log({'IoU', {n: 100 * v for n, v in zip(metrics.name_classes, metrics.IoU())}, step})

            torch.save(model.state_dict(), logdir + "/latest.pth")

        ########################
        # VALIDATION
        miou = validate(writer, vset, vloader, e, model, device)
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), logdir + "/val_best.pth")


if __name__ == "__main__":
    args = {"seed": 12345}
    set_seed(args["seed"])

    trainer = Trainer(args)
    trainer.train()

    trainer.writer.flush()
    trainer.writer.close()
