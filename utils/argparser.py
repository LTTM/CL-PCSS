import argparse
import os
import shutil
from tensorboardX import SummaryWriter
import logging

from dataloaders.PCSdataset import PCSDataset
from dataloaders.S3DISdataset import S3DISDataset
from dataloaders.Semantic3Ddataset import Semantic3DDataset
from dataloaders.SemanticKITTIdataset import SemanticKITTIDataset
#from dataloaders.SELMAdataset import SELMADataset

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError("Unrecognized boolean: "+v)

def str2str_none_num(v,t=float):
    if v.lower() in ('none',):
        return None
    else:
        try:
            return t(v)
        except ValueError:
            return v

def str2intlist(v):
    l = v.split(',')
    l = [str2str_none_num(el,t=int) for el in l]
    return l if len(l)>1 else l[0]

def str2floatlist(v):
    l = v.split(',')
    l = [str2str_none_num(el,t=float) for el in l]
    return l if len(l)>1 else l[0]

def parse_dataset(dname):
    if dname=='sem3D':
        return Semantic3DDataset
    #elif dname=='selma':
    #    return SELMADataset
    elif dname=='kitti':
        return SemanticKITTIDataset
    elif dname=='s3dis':
        return S3DISDataset
    else:
        return PCSDataset

def init_params(train_type='train'):

    argparser = argparse.ArgumentParser()

    if train_type in ['train', 'eval']:    
        argparser.add_argument('--dataset', default="kitti", type=parse_dataset,
                               choices=[Semantic3DDataset, SemanticKITTIDataset, S3DISDataset, PCSDataset], #SELMADataset, 
                               help="The dataset used for supervised training, choose from ['selma', 'sem3d', 'kitti', 's3dis', 'pcs']")                              


    if train_type in ['test']:
        argparser.add_argument('--test_split', default='test', type=str,
                               help='Split file to be used for test samples')

    if train_type in ['train', 'eval']:
        argparser.add_argument('--augment_data', default=True, type=str2bool,
                               help='Whether to augment the (training) images with flip & Gaussian Blur')
                               
        argparser.add_argument('--random_noise', default=True, type=str2bool,
                               help='Whether to add random noise')
        argparser.add_argument('--random_rot', default=True, type=str2bool,
                               help='Whether to randomly rotate point clouds')
        argparser.add_argument('--random_shift', default=True, type=str2bool,
                               help='Whether to randomly shift point clouds')
        argparser.add_argument('--random_res_crop', default=True, type=str2bool,
                               help='Whether to randomly crop and rescale parts of the point clouds')

        argparser.add_argument('--batch_size', default=1, type=int, #2
                               help='Training batch size')
        argparser.add_argument('--val_batch_size', default=1, type=int,
                               help='Validation batch size')
        argparser.add_argument('--dataloader_workers', default=2, type=int, #4
                               help='Number of workers to use for each dataloader (significantly affects RAM consumption)')

        argparser.add_argument('--classifier', default='RandLA-Net', type=str,
                               choices=['ESegCloud', 'RandLA-Net', 'Cylinder3D', 'Rangenet++'],
                               help='Which classifier head to use in the model')
        argparser.add_argument('--seed', default=12345, type=int,
                                   help='Seed for the RNGs, for repeatability')
   
    if train_type in ['train']:
        argparser.add_argument('--sup_loss', default='ce', type=str, choices=['ce', 'hnmce'],
                               help='The supervised loss to be used for optimization')
        argparser.add_argument('--lr', default=1e-3, type=float,
                               help='The initial learning rate to be used')
        argparser.add_argument('--lre', default=1e-4, type=float,
                               help='The final learning rate to be used')
        argparser.add_argument('--poly_power', default=.9, type=float,
                                   help='lr polynomial decay rate')
        argparser.add_argument('--decay_over_iterations', default=1000, type=int,
                               help='lr polynomial decay max_steps')
        argparser.add_argument('--epochs', default=1, type=int,
                               help='Number of iterations performed')
        argparser.add_argument('--momentum', default=.9, type=float,
                                   help='SGD optimizer momentum')
        argparser.add_argument('--weight_decay', default=1e-5, type=float,
                               help='SGD optimizer weight decay')
        argparser.add_argument('--eval_every_n_epochs', default=1, type=int,
                               help='Number of iterations every which a validation is run, <= 0 disables validation')
        argparser.add_argument('--ce_use_weights', default=False, type=str2bool,
                               help='Whether to use pixel frequencies to normalize the cross-entropy')
        """
        # LEAK
        argparser.add_argument('--LEAK', default=False, type=str2bool,
                               help='Whether to use LEAK regulariation methods')
        argparser.add_argument('--LEAK_fair', default=10., type=float,
                               help='LEAK attentive fair regualization weight')
        argparser.add_argument('--LEAK_proto', default=0.1, type=float,
                               help='LEAK proto-feat alignmen regualization weight')
        argparser.add_argument('--LEAK_proto_macro', default=0.001, type=float,
                               help='LEAK macro proto-feat alignment regualization weight')
        """
    argparser.add_argument('--logdir', default="log", type=str, #/%d"%(int(time.time()))
                   help='Path to the log directory')
    argparser.add_argument('--test_name', default=None, type=str,  # /%d"%(int(time.time()))
                           help='Test name')
    argparser.add_argument('--pretrained_model', default=False, type=str2bool,
                           help='Whether to load pretrained model')
    argparser.add_argument('--ckpt_file', default="log/train/CIL/val_best.pth", type=str,
                   help='Path to the model checkpoint')

    """
    argparser.add_argument('--logger', default='wandb', type=str, choices=['wandb', 'tensorboard'],
                           help='Selected logger')
    argparser.add_argument('--proto_plot', default=False, type=str2bool,
                   help='Whether to plot protoypes, K')
    argparser.add_argument('--tsne_plot', default=False, type=str2bool,
                   help='Whether to plot tsne in validation')
    argparser.add_argument('--pcs_plot', default=False, type=str2bool,
                   help='Whether to plot point clouds in validation')
    argparser.add_argument('--analyses', default=False, type=str2bool,
                   help='Whether to compute ccd and ipd')
    """

    #### CONTINUAL ###
    argparser.add_argument('--CL', default=False, type=str2bool,
                           help='Whether to apply a continual learning setup or not')
    argparser.add_argument('--method', default='CIL', type=str, choices=['CIL', 'Priori', 'SC', 'Random'],
                           help='Selected dset subdivision method for CL')
    argparser.add_argument('--c2f', default=False, type=str2bool,
                           help='Whather c2f is enabled')
    argparser.add_argument('--setup', default='Sequential', type=str, choices=['Sequential_masked','Sequential','Disjoint','Overlapped'],
                           help='Selected setup for CL')
    argparser.add_argument('--CL_steps', default=3, type=int,
                           help='Number of steps for CL')

    return argparser.parse_args()
    
class StripColorsFormatter(logging.Formatter):
    def format(self, record):
        fmt = super(StripColorsFormatter, self).format(record)
        return fmt.replace('\033[91m','').replace('\033[92m','').replace('\033[93m','').replace('\033[96m','').replace('\033[0m','')
    
def init_logger(args):
    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)

    # create the log path
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(args.logdir, flush_secs=.5)

    # logger configure
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.logdir, 'train_log.txt'))
    ch = logging.StreamHandler()
    fhformatter = StripColorsFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    chformatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fhformatter)
    ch.setFormatter(chformatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Global configuration as follows:")
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
        
    return writer, logger

#function to print configuration parameters
def print_cfg(cfg):
    out_string = ""
    varconf = vars(cfg)
    for key in varconf:
        if key.startswith("__"):
            continue
        elif key=="dataset":
            out_string += key + ": " + str(varconf[key].name) + '\n'
        elif key=="lr_decays":
            l = len(varconf[key])
            out_string += key +": "+str(varconf[key][l-1])+" per epoch, for "+str(l)+" epochs" + '\n'
        else:
            out_string += key +": "+str(varconf[key]) + '\n'
    return out_string