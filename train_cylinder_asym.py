# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py

import os
import time
import argparse
import sys

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings, yaml
import LEAK
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")


def tsne(features, labels, prototypes, lab=None):  # works
    B = features.shape[0]  # batch size
    depth = features.shape[1]  # length of feature vectors
    a, b, c = features.shape[2:]
    features = torch.swapaxes(features, 0, 1)
    xx = torch.reshape(labels, (B * a * b * c,)).long().detach().cpu().numpy()
    mask = (xx!=0)
    xx = xx[mask]
    yy = torch.reshape(features, (depth, B * a * b * c))
    yy = torch.swapaxes(yy, 0, 1).detach().cpu().numpy() # n*depth
    yy = yy[mask]
    wandb.log({"image": wandb.Image(yy, caption="feats")})
    yy = np.vstack([yy, prototypes.detach().cpu().numpy()])
    xx = np.hstack([xx, np.arange(0,20)])
    tsne = TSNE(n_components=2, random_state=0, learning_rate=10, init='pca', perplexity=100, n_iter=1000)
    return tsne.fit_transform(yy), xx

def tsne_plot(Y, labs): #works -> wandb log
    classes = np.unique(labs)
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
    cmap=cmap/255.0
    fig, ax = plt.subplots()
    for c in classes:
        cl = np.hstack(labs == c)
        ax.scatter(Y[cl, 0], Y[cl, 1], color=cmap[c], s=0.1)
        ax.scatter(Y[cl[-c], 0], Y[cl[-c], 1], marker="*", color=cmap[-c], s=12)
    plt.savefig('img.png')

def main(args):
    wandb.init(project="LEAK-Elena", entity="elena_camuffo")
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    wandb.config = {"fair": args.LEAK_fair,
                  "proto": args.LEAK_proto,
                  "m-proto": args.LEAK_proto_macro}

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)
                                                   
    data_config = os.path.join('utils', 'LEAK_Utils.yaml')
    DATA = yaml.safe_load(open(data_config, 'r'))

    #fair_loss = LEAK.FairnessLoss(alpha=args.LEAK_fair, classes=num_class)
    fair_loss = LEAK.FairLoss(alpha=args.LEAK_fair, mapping=DATA["micro2macro_SemanticKITTI"], idx=DATA["macro_idx_SemanticKITTI"], classes=num_class)
    proto_loss = LEAK.ProtoLoss(ignore_index=-1, gamma=args.LEAK_proto, classes=num_class)
    macro_proto_loss = LEAK.ProtoLoss(ignore_index=-1, gamma=args.LEAK_proto_macro, classes=num_class)
    prototypes = torch.zeros((num_class, 512)).cuda()
    macro_prototypes = torch.zeros((len(DATA["macro_idx_SemanticKITTI"]), 512)).cuda()
    K = torch.zeros((num_class)).cuda()
    M = torch.zeros((len(DATA["macro_idx_SemanticKITTI"]))).cuda()

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config, train_dataloader_config, val_dataloader_config, grid_size=grid_size)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    #tsne_f = []
    #tsne_l = []

    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)
        # lr_scheduler.step(epoch)
        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
            if global_iter % check_iter == 0: # and epoch >= 1:
                my_model.eval()
                hist_list = []
                val_loss_list = []
                vbar = tqdm(total=len(val_dataset_loader))
                with torch.no_grad():
                    for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(
                            val_dataset_loader):

                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                        val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

                        predict_labels, feats = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)

                        def get_class_weights(dataset_name):
                            # pre-calculate the number of points in each category
                            num_per_class = []
                            if dataset_name is 'S3DIS':
                                num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                                          650464, 791496, 88727, 1284130, 229758, 2272837],
                                                         dtype=np.int32)
                            elif dataset_name is 'Semantic3D':
                                num_per_class = np.array(
                                    [5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                    dtype=np.int32)
                            elif dataset_name is 'SemanticKITTI':
                                num_per_class = np.array(
                                    [55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                     240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                     9833174, 129609852, 4506626, 1168181])
                            weight = num_per_class / float(sum(num_per_class))
                            ce_label_weight = 1 / np.sqrt(weight + 0.02)
                            return np.expand_dims(ce_label_weight, axis=0)

                        # aux_loss = loss_fun(aux_outputs, point_label_tensor)
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                              ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)

                        # if leak
                        point_label_tensor = val_label_tensor
                        q = int(point_label_tensor.shape[1] / feats.shape[2])
                        w = int(point_label_tensor.shape[2] / feats.shape[3])
                        h = int(point_label_tensor.shape[3] / feats.shape[4])
                        fy = torch.zeros(point_label_tensor[:, ::q, ::w, ::h].shape)
                        fmy = torch.zeros_like(fy)

                        for a in range(0, point_label_tensor.shape[0]):
                            for b in range(q, point_label_tensor.shape[1], q):
                                for c in range(w, point_label_tensor.shape[2], w):
                                    for d in range(h, point_label_tensor.shape[3], h):
                                        sub_tensor = val_vox_label[a, b - q:b, c - w:c, d - h:d]
                                        u, uc = torch.unique(sub_tensor, return_counts=True)
                                        if len(u) > 1:
                                            uc_max = torch.argmax(uc[1:]) + 1
                                        else:
                                            uc_max = 0
                                        fy[a, int(b / q), int(c / w), int(d / w)] = int(u[uc_max])
                                        fmy[a, int(b / q), int(c / w), int(d / w)] = DATA["micro2macro_SemanticKITTI"][
                                            int(u[uc_max])]

                        #if i_iter_val % 1000 == 0:
                         #   f, l = tsne(feats, fy, prototypes)
                         #   tsne_plot(f,l)
                         #   img = cv2.imread('img.png')
                         #   wandb.log({"image":wandb.Image(img, caption="val image")})
                         #   wandb.log({"image": wandb.Image(prototypes.detach().cpu().numpy(), caption="protos")})

                        loss += fair_loss(predict_labels, point_label_tensor)
                        prototypes, K, macro_prototypes, M = LEAK.build_prototypes(feats, fy, prototypes, K,
                                                                                   fmy, macro_prototypes, M)
                        loss += proto_loss(feats, fy, prototypes)  # micro
                        loss += macro_proto_loss(feats, fmy, macro_prototypes)  # macro

                        predict_labels = torch.argmax(predict_labels, dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        for count, i_val_grid in enumerate(val_grid):
                            hist_list.append(fast_hist_crop(predict_labels[
                                                                count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                                val_grid[count][:, 2]], val_pt_labs[count],
                                                            unique_label))
                        val_loss_list.append(loss.detach().cpu().numpy())
                        vbar.update(1)
                    vbar.close()
                my_model.train()
                iou = per_class_iu(sum(hist_list))
                print('Validation per class iou: ')
                for class_name, class_iou in zip(unique_label_str, iou):
                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                val_miou = np.nanmean(iou) * 100
                del val_vox_label, val_grid, val_pt_fea, val_grid_ten
                wandb.log({"val_miou": val_miou})

                # save model if performance is improved
                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    torch.save(my_model.state_dict(), model_save_path)

                print('Current val miou is %.3f while the best val miou is %.3f' %
                      (val_miou, best_val_miou))
                print('Current val loss is %.3f' %
                    (np.mean(val_loss_list)))

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)

            # forward + backward + optimize
            outputs, feats = my_model(train_pt_fea_ten, train_vox_ten, train_batch_size)
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=0) + loss_func(
                outputs, point_label_tensor)

            #if leak
            q = int(point_label_tensor.shape[1]/feats.shape[2])
            w = int(point_label_tensor.shape[2]/feats.shape[3])
            h = int(point_label_tensor.shape[3]/feats.shape[4])
            fy = torch.zeros(point_label_tensor[:, ::q, ::w, ::h].shape)
            fmy = torch.zeros_like(fy)

            for a in range(0, point_label_tensor.shape[0]):
                for b in range(q, point_label_tensor.shape[1], q):
                    for c in range(w, point_label_tensor.shape[2], w):
                        for d in range(h, point_label_tensor.shape[3], h):
                            sub_tensor = train_vox_label[a, b - q:b, c - w:c, d - h:d]
                            u, uc = torch.unique(sub_tensor, return_counts=True)
                            if len(u) > 1:
                                uc_max = torch.argmax(uc[1:]) + 1
                            else:
                                uc_max = 0
                            fy[a, int(b / q), int(c / w), int(d / w)] = int(u[uc_max])
                            fmy[a, int(b / q), int(c / w), int(d / w)] = DATA["micro2macro_SemanticKITTI"][
                                int(u[uc_max])]

            # f = np.swapaxes(feats, 0, 1)
            # f = f.reshape(15 * 12 * 4 * 2, f.shape[0])
            # l = fy.reshape(15 * 12 * 4 * 2)
            # mask = l != 0
            # f = f[mask]
            # l = l[mask]
            # tsne_f.extend(feats)
            # tsne_l.extend(fy)
            if i_iter%1000==0:
                f, l = tsne(feats, fy, prototypes)
                tsne_plot(f, l)
                img = cv2.imread('img.png')
                wandb.log({"image":wandb.Image(img, caption="train image")})

            if args.LEAK:
                train_fair_loss = fair_loss(outputs, point_label_tensor)
                loss += train_fair_loss
                prototypes, K, macro_prototypes, M = LEAK.build_prototypes(feats, fy, prototypes, K, fmy,
                                                                           macro_prototypes, M)
                train_p_loss = proto_loss(feats, fy, prototypes)
                train_mp_loss = macro_proto_loss(feats, fmy, macro_prototypes)
                loss += train_p_loss        # micro
                loss += train_mp_loss       # macro

                wandb.log({"fair_loss": train_fair_loss})
                wandb.log({"macro_proto_loss": train_mp_loss})
                wandb.log({"proto_loss": train_p_loss})

            wandb.log({"train_loss": loss})

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')
        pbar.close()
        epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    parser.add_argument('--LEAK', default=True, type=bool,
                        help='Whether to use LEAK')
    parser.add_argument('--LEAK_fair', default=0., type=float,
                               help='LEAK attentive fair regualization weight')
    parser.add_argument('--LEAK_proto', default=100., type=float,
                               help='LEAK prototypes weight')
    parser.add_argument('--LEAK_proto_macro', default=1., type=float,
                               help='LEAK macro prototypes weight')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
