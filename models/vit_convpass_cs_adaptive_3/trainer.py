import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.utils import AverageMeter
from .network import build_net

pd.set_option('display.max_columns', None)

import logging

from data.transforms import VisualTransform, get_augmentation_transforms

from test import metric_report_from_dict

# try:
# from torch.utils.tensorboard import SummaryWriter
# except:
from tensorboardX import SummaryWriter

from FAS_DataManager.get_dataset import get_image_dataset_from_list

class Trainer():
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config):
        """
        Construct a new Trainer instance.
        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """

        self.config = config
        self.global_step = 1
        self.start_epoch = 1

        # Training control config
        self.epochs = self.config.TRAIN.EPOCHS
        self.batch_size = self.config.DATA.BATCH_SIZE

        self.counter = 0

        #  # Meanless at this version
        self.epochs = self.config.TRAIN.EPOCHS
        self.val_freq = config.TRAIN.VAL_FREQ
        #
        # # Network config

        self.val_metrcis = {
            'HTER@0.5': 1.0,
            'EER': 1.0,
            'MIN_HTER': 1.0,
            'AUC': 0
        }

        # Optimizer config
        self.momentum = self.config.TRAIN.MOMENTUM
        self.init_lr = self.config.TRAIN.INIT_LR
        self.lr_patience = self.config.TRAIN.LR_PATIENCE
        self.train_patience = self.config.TRAIN.PATIENCE


        self.train_mode = True

        kwargs = {
            'conv_type': self.config.MODEL.CONV,
            'num_classes': self.config.MODEL.NUM_CLASSES
        }
        self.network = build_net(arch_name=config.MODEL.ARCH, pretrained=config.MODEL.IMAGENET_PRETRAIN, **kwargs)
        if self.config.CUDA:
            self.network.cuda()

        self.loss = torch.nn.CrossEntropyLoss()

    def init_weight(self, ckpt_path):
        logging.info("[*] Initialize model weight from {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        # self.best_valid_acc = ckpt['best_valid_acc']
        self.network.load_state_dict(ckpt['model_state'])

    def get_dataset(self, data_file_list, transform):
        datasets = []
        for data_list_path in data_file_list:
            datasets.append(get_image_dataset_from_list(data_list_path, transform))
        return torch.utils.data.ConcatDataset(datasets)



    def get_dataloader(self):
        config = self.config

        train_batch_size = config.DATA.BATCH_SIZE
        val_batch_size = config.TEST.BATCH_SIZE
        test_batch_size = config.TEST.BATCH_SIZE
        num_workers = 0 if config.DEBUG else config.DATA.NUM_WORKERS

        dataset_root_dir = config.DATA.ROOT_DIR
        dataset_subdir = config.DATA.SUB_DIR  # 'EXT0.2'
        dataset_dir = os.path.join(dataset_root_dir, dataset_subdir)

        test_data_transform = VisualTransform(config)

        if not self.train_mode:
            assert config.DATA.TEST, "Please provide at least a data_list"
            test_dataset = self.get_dataset(config.DATA.TEST, test_data_transform)
            self.test_data_loader = torch.utils.data.DataLoader(test_dataset, test_batch_size, num_workers=num_workers,
                                                                shuffle=False, drop_last=True)
            return self.test_data_loader

        else:
            assert config.DATA.TRAIN, "CONFIG.DATA.TRAIN should be provided"
            aug_transform = get_augmentation_transforms(config)
            train_data_transform = VisualTransform(config, aug_transform)
            train_dataset = self.get_dataset(config.DATA.TRAIN, train_data_transform)
            self.train_data_loader = torch.utils.data.DataLoader(train_dataset, train_batch_size, num_workers=num_workers,
                                                                 shuffle=True, pin_memory=True, drop_last=True)

            assert config.DATA.VAL, "CONFIG.DATA.VAL should be provided"
            val_dataset = self.get_dataset(config.DATA.VAL, test_data_transform)
            self.val_data_loader = torch.utils.data.DataLoader(val_dataset, val_batch_size, num_workers=num_workers,
                                                               shuffle=False, pin_memory=True, drop_last=True)

            return self.train_data_loader, self.val_data_loader
    def train(self, ):

        if self.config.TRAIN.INIT and os.path.exists(self.config.TRAIN.INIT):
            self.init_weight(self.config.TRAIN.INIT)

        if self.config.MODEL.FIX_BACKBONE:
            for name, p in self.network.named_parameters():
                if 'adapter' in name or 'head' in name:
                    p.requires_grad = True
                    # import pdb; pdb.set_trace()
                else:
                    p.requires_grad = False
        # Set up optimizer
        if self.config.TRAIN.OPTIM.TYPE == 'SGD':
            logging.info('Setting: Using SGD Optimizer')
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.network.parameters()),
                lr=self.init_lr,
            )

        elif self.config.TRAIN.OPTIM.TYPE == 'Adam':
            logging.info('Setting: Using Adam Optimizer')
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.network.parameters()),
                lr=self.init_lr,
            )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.TRAIN.EPOCHS)

        # Set up data
        train_data_loader = self.train_data_loader
        val_data_loader = self.val_data_loader
        tensorboard_dir = os.path.join(self.config.OUTPUT_DIR, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tensorboard = None if self.config.DEBUG else SummaryWriter(tensorboard_dir)

        self.num_train = len(train_data_loader) * self.config.DATA.BATCH_SIZE
        self.num_valid = len(val_data_loader) * self.config.DATA.BATCH_SIZE
        logging.info("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid))

        if self.config.TRAIN.RESUME and os.path.exists(self.config.TRAIN.RESUME):
            logging.info("Resume=True.")
            self.load_checkpoint(self.config.TRAIN.RESUME)

        if self.config.CUDA:
            logging.info("Number of GPUs: {}".format(torch.cuda.device_count()))
            self.network = torch.nn.DataParallel(self.network)

        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.tensorboard:
                self.tensorboard.add_scalar('lr', self.init_lr, self.global_step)
            logging.info('\nEpoch: {}/{} - LR: {:.6f}'.format(
                epoch, self.epochs, self.init_lr))

            # train for 1 epoch
            train_loss = AverageMeter()
            self.network.train()
            num_train = len(train_data_loader) * self.batch_size

            with tqdm(total=num_train, ncols=80) as pbar:
                for i, batch_data in enumerate(train_data_loader):

                    loss = self._train_one_batch(batch_data=batch_data,
                                                 optimizer=self.optimizer
                                                 )

                    pbar.set_description(
                        (
                            " total loss={:.3f} ".format(loss.item(),
                                                         )
                        )
                    )
                    pbar.update(self.batch_size)
                    train_loss.update(loss.item(), self.batch_size)
                    # log to tensorboard
                    if self.tensorboard:
                        self.tensorboard.add_scalar('loss/train_total', loss.item(), self.global_step)

                    self.global_step += 1

                self.lr_scheduler.step()

            train_loss_avg = train_loss.avg

            logging.info("Avg Training loss = {} ".format(str(train_loss_avg)))
            # evaluate on validation set'
            if epoch % self.val_freq == 0:
                with torch.no_grad():
                    val_output = self.validate(epoch, val_data_loader)

                if val_output['MIN_HTER'] < self.val_metrcis['MIN_HTER']:
                    logging.info("Save models")
                    self.val_metrcis['MIN_HTER'] = val_output['MIN_HTER']
                    self.val_metrcis['AUC'] = val_output['AUC']
                    self.save_checkpoint(
                        {'epoch': epoch,
                         'val_metrics': self.val_metrcis,
                         'global_step': self.global_step,
                         'model_state': self.network.module.state_dict(),
                         'optim_state': self.optimizer.state_dict(),
                         }
                    )

                logging.info('Current Best MIN_HTER={}%, AUC={}%'.format(100 * self.val_metrcis['MIN_HTER'],
                                                                         100 * self.val_metrcis['AUC']))

    def _train_one_batch(self, batch_data, optimizer):
        network_input, target = batch_data[1], batch_data[2]

        cls_out = self.inference(network_input.cuda())
        # compute losses for differentiable modules

        loss = self._total_loss_caculation(cls_out, target)

        # compute gradients and update SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def validate(self, epoch, val_data_loader):
        val_results = self.test(val_data_loader)
        val_loss = val_results['avg_loss']
        scores_gt_dict = val_results['scores_gt']
        scores_pred_dict = val_results['scores_pred']
        # log to tensorboard
        if self.tensorboard:
            self.tensorboard.add_scalar('loss/val_total', val_loss, self.global_step)

        frame_metric_dict, video_metric_dict = metric_report_from_dict(scores_pred_dict, scores_gt_dict, 0.5)
        df_frame = pd.DataFrame(frame_metric_dict, index=[0])
        df_video = pd.DataFrame(video_metric_dict, index=[0])

        logging.info("Frame level metrics: \n" + str(df_frame))
        logging.info("Video level metrics: \n" + str(df_video))

        return frame_metric_dict

    def test(self, test_data_loader):
        avg_test_loss = AverageMeter()
        scores_pred_dict = {}
        spoofing_label_gt_dict = {}
        self.network.eval()
        with torch.no_grad():
            for data in tqdm(test_data_loader, ncols=80):
                network_input, target, video_ids = data[1], data[2], data[3]

                output_prob = self.inference(network_input.cuda())
                test_loss = self._total_loss_caculation(output_prob, target)
                pred_score = self._get_score_from_prob(output_prob)
                avg_test_loss.update(test_loss.item(), network_input.size()[0])

                gt_dict, pred_dict = self._collect_scores_from_loader(spoofing_label_gt_dict, scores_pred_dict,
                                                                      target['spoofing_label'].numpy(), pred_score,
                                                                      video_ids
                                                                      )

        test_results = {
            'scores_gt': gt_dict,
            'scores_pred': pred_dict,
            'avg_loss': avg_test_loss.avg,
        }
        return test_results

    def _collect_scores_from_loader(self, gt_dict, pred_dict, ground_truths, pred_scores, video_ids):
        batch_size = ground_truths.shape[0]

        for i in range(batch_size):
            video_name = video_ids[i]
            if video_name not in pred_dict.keys():
                pred_dict[video_name] = list()
            if video_name not in gt_dict.keys():
                gt_dict[video_name] = list()

            pred_dict[video_name] = np.append(pred_dict[video_name], pred_scores[i])
            gt_dict[video_name] = np.append(gt_dict[video_name], ground_truths[i])

        return gt_dict, pred_dict

    def save_checkpoint(self, state):

        ckpt_dir = os.path.join(self.config.OUTPUT_DIR, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)

        epoch = state['epoch']
        if self.config.TRAIN.SAVE_BEST:
            filename = 'best.ckpt'.format(epoch)
        else:
            filename = 'epoch_{}.ckpt'.format(epoch)
        ckpt_path = os.path.join(ckpt_dir, filename)
        logging.info("[*] Saving model to {}".format(ckpt_path))
        torch.save(state, ckpt_path)

    def load_checkpoint(self, ckpt_path):

        logging.info("[*] Loading model from {}".format(ckpt_path))

        ckpt = torch.load(ckpt_path)
        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.valid_metric = ckpt['val_metrics']
        # self.best_valid_acc = ckpt['best_valid_acc']
        self.network.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        logging.info(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                ckpt_path, ckpt['epoch'])
        )

    def inference(self, *args, **kargs):
        """
            Input images
            Output prob and scores
        """
        output_prob = self.network(*args, **kargs)  # By default: a binary classifier network
        return output_prob

    def _total_loss_caculation(self, output_prob, target):
        spoofing_label = target['spoofing_label'].cuda()
        return self.loss(output_prob, spoofing_label)

    def _get_score_from_prob(self, output_prob):
        output_scores = torch.softmax(output_prob, 1)
        output_scores = output_scores.cpu().numpy()[:, 1]
        return output_scores

    def load_batch_data(self):
        pass
