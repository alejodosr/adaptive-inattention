from pytorch_lightning import Trainer
import torch
from models.inatt_models_ptl import HRMobileVOD
from models.backbone import SSD, MobileNetV1, MobileVOD
from config import mobilenetv1_ssd_config
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.misc import str2bool, Timer, store_labels
from dataloaders.vid_dataset import ImagenetDataset, VIDDataset
from dataloaders.voc_dataset import VOCDataset
from dataloaders.yolo_dataset import YOLODataset, YOLODatasetVID
from dataloaders.data_preprocessing import TrainTransform, ValTransform
from models.backbone import MatchPrior
import os
import sys
import argparse
import logging

# Argument parser
parser = argparse.ArgumentParser(description='High-Resolution Mobile Video Object Detection (Bottleneck LSTM) Training With Pytorch Lightning')

parser.add_argument('--datasets', type=str, help='Dataset directory path')
parser.add_argument('--val_dataset', type=str, default='', help='Dataset directory path')
parser.add_argument('--cache_path', type=str, help='Cache directory path')
parser.add_argument('--width_mult', default=1.0, type=float,
                    help='Width Multiplifier')
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")
parser.add_argument('--sequence_length', default=10, type=int,
                    help='sequence_length of video to unfold')
parser.add_argument("--dataset_type", default="imagenet_vid", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=0, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update')
# parser.add_argument('--base_net_lr', default=None, type=float,
#                     help='initial learning rate for base net.')
# parser.add_argument('--ssd_lr', default=None, type=float,
#                     help='initial learning rate for the layers not in base net')

# Params for loading pretrained backbone or checkpoints.
parser.add_argument('--pretrained', default='', type=str, help='Pre-trained model')
parser.add_argument('--resume', default='', type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="plateau", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Params for Exponential Lr decay
parser.add_argument('--gamma_sched', default=0.95, type=float,
                    help='gamma_sched value for Exponential Lr decay Scheduler.')

# Train params
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=200, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=3, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--gpu_id', default=0, type=int,
                    help='The GPU id to be used')
parser.add_argument('--num_classes', default=30, type=int,
                    help='')
parser.add_argument('--checkpoint_folder', default='checkpoints/', type=str,
                    help='Directory for saving checkpoint models')
parser.add_argument('--net', default="lstm9", type=str,
                    help="The network architecture, it should be of backbone, lstm.")
parser.add_argument('--backbone', default="mobilenetv1", type=str,
                    help="The feature extraction backbone, it should be mobilenetv1 or mobilenetv2.")
parser.add_argument('--crop_prob', default=0, type=float,
                    help='Probability of crops during training, only used in lstm training')
parser.add_argument('--bbox_increase_factor', default=0.1, type=float,
                    help='Increase factor of crops when using them')



args = parser.parse_args()

# Initialize logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Check if CUDA is available
if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")

def print_dict(dict):
    for k, v in dict.items():
        print(k)

def initialize_model(net):
    """ Loads learned weights from pretrained checkpoint model
    Arguments:
        net : object of MobileVOD
    """
    if args.pretrained:
        logging.info(f"Loading weights from pretrained mobilenetv1 netwok ({args.pretrained})")
        pretrained_net_dict = torch.load(args.pretrained)['state_dict']
        model_dict = net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_net_dict = {k: v for k, v in pretrained_net_dict.items() if
                               k in model_dict and model_dict[k].shape == pretrained_net_dict[k].shape}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_net_dict)
        # 3. load the new state dict
        net.load_state_dict(model_dict)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Initialize timer
    timer = Timer()

    logging.info("Build network.")

    # Transforms
    if args.net == 'backbone':
        config = mobilenetv1_ssd_config
        train_transform = TrainTransform(config.image_size, config.image_mean, config.image_std)
        target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
        val_transform = ValTransform(config.image_size, config.image_mean, config.image_std)

        # Datasets
        if args.dataset_type == 'imagenet_vid':
            val_dataset = ImagenetDataset(args.datasets, args.cache_path, transform=val_transform,
                                          target_transform=target_transform, is_val=True)
            train_dataset = ImagenetDataset(args.datasets, args.cache_path, transform=train_transform,
                                            target_transform=target_transform)
        elif args.dataset_type == 'voc':
            if args.val_dataset is not '':
                val_dataset = VOCDataset(args.val_dataset, transform=val_transform, is_test=True,
                                              target_transform=target_transform)
            else:
                val_dataset = VOCDataset(args.datasets,  transform=val_transform, is_test=True,
                                              target_transform=target_transform)
            train_dataset = VOCDataset(args.datasets, transform=train_transform,
                                            target_transform=target_transform)
        elif args.dataset_type == 'yolo':
            val_dataset = YOLODataset(args.datasets, transform=val_transform, is_val=True,
                                              target_transform=target_transform)

            train_dataset = YOLODataset(args.datasets, transform=train_transform,
                                            target_transform=target_transform)
        else:
            logging.fatal("Train dataset not found")
            parser.print_help(sys.stderr)
            sys.exit(1)

        # Custom learning rate
        # args.lr = 1e-3
    elif args.net == 'lstm':
        config = mobilenetv1_ssd_config
        train_transform = TrainTransform(config.image_size, config.image_mean, config.image_std)
        target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
        val_transform = ValTransform(config.image_size, config.image_mean, config.image_std)

        # Datasets
        if args.dataset_type == 'imagenet_vid':
            if args.val_dataset != '':
                val_dataset = ImagenetDataset(args.val_dataset, args.cache_path, transform=val_transform,
                                              target_transform=target_transform, is_val=True)
            else:
                val_dataset = ImagenetDataset(args.datasets, args.cache_path, transform=val_transform,
                                              target_transform=target_transform, is_val=True)
            train_dataset = VIDDataset(args.datasets, args.cache_path, transform=train_transform,
                                            target_transform=target_transform, batch_size=args.batch_size)
        elif args.dataset_type == 'yolo':
            if args.val_dataset != '':
                val_dataset = YOLODataset(args.val_dataset, transform=val_transform, is_val=True,
                                          target_transform=target_transform)
            else:
                val_dataset = YOLODataset(args.datasets, transform=val_transform, is_val=True,
                                          target_transform=target_transform)
            train_dataset = YOLODatasetVID(args.datasets, transform=train_transform,
                                            target_transform=target_transform, batch_size=args.batch_size)
        else:
            logging.fatal("Val dataset not found")
            parser.print_help(sys.stderr)
            sys.exit(1)

        # Custom learning rate
        args.lr = 1e-4
    else:
        logging.fatal("The net type is wrong. It should be one of backbone, lstm.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    args.num_classes = int(len(val_dataset._classes_names))
    logging.info(f"Number of classes: {args.num_classes}")

    model = HRMobileVOD(args, logging=logging)

    model.set_val_dataset(val_dataset)
    model.set_train_dataset(train_dataset)
    model.set_cache_dir(args.cache_path)

    # Initialize with pretrained
    initialize_model(model)

    print("The number of parameters of " + args.net + " is " + str(count_parameters(model)))

    # If set to freeze backbone weights
    if args.freeze_net:
        logging.info("Freeze net.")
        for param in model.pred_enc.parameters():
            param.requires_grad = False
        model.pred_dec.conv13.requires_grad = False

    # Trainer
    saving_directory = os.path.join(os.path.join(os.getcwd(), args.checkpoint_folder), args.net)
    logging.info(f"Checkpoints saving directory: {saving_directory}.")

    checkpoint_callback = ModelCheckpoint(
        filepath=saving_directory,
        save_top_k=True,
        verbose=True,
        monitor='map',
        mode='max',
        prefix=''
    )

    trainer = Trainer(max_nb_epochs=args.num_epochs,
                      gpus=[args.gpu_id],
                      train_percent_check=1.0,
                      check_val_every_n_epoch=args.validation_epochs,
                      val_percent_check=1.0,
                      show_progress_bar=True,
                      resume_from_checkpoint=args.resume if args.resume != '' else None,
                      default_save_path=saving_directory,
                      checkpoint_callback=checkpoint_callback)

    # Store labels
    label_file = os.path.join(args.checkpoint_folder, "vid-model-labels.txt")
    store_labels(label_file, model.val_dataset._classes_names)
    logging.info(f"Stored labels into file {label_file}.")

    logging.info("Train dataset size: {}".format(len(model.train_dataset)))
    logging.info("validation dataset size: {}".format(len(model.val_dataset)))

    # view tensorflow logs
    logging.info(f'View tensorboard logs by running\ntensorboard --logdir {os.getcwd()}')
    logging.info('and going to http://localhost:6006 on your browser')

    # train
    trainer.fit(model)


