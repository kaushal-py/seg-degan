'''
System for semantic segmentation
Author: Kaushal Bhogale (kaushal-py)
License: MIT

Uses deeplabv3 architecture to perform semantic segmentation.
Provides utility for 3 datasets:
    - Camvid
    - Cityscapes
    -NYUv2
The backbone architectures supported are:
    - Resnet50 pretrained
    - Resnet50
    - MobilenetV2
'''

# Imports
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

from code.dataset.camvid import CamVid
from code.dataset.cityscapes import Cityscapes
from code.dataset.nyu import NYUv2
import code.network.segmentation.deeplabv3 as deeplabv3
import code.utils as utils


class SegmentationSystem:
    '''
    Semantic segmentation module
    Pass configurations and hyperparameters from a trainer file.
    '''
    def __init__(self, config, hparams):

        # Store config and hparams
        self.config = config
        self.hparams = hparams

        # Camvid Dataset
        if self.config.dataset == 'CamVid':
            train_transforms = utils.ext_transforms.ExtCompose([
                utils.ext_transforms.ExtResize(256),
                utils.ext_transforms.ExtRandomCrop(128, pad_if_needed=True),
                utils.ext_transforms.ExtRandomHorizontalFlip(),
                utils.ext_transforms.ExtToTensor(),
                utils.ext_transforms.ExtNormalize((0.5, ), (0.5, )),
            ])
            test_transforms = utils.ext_transforms.ExtCompose([
                utils.ext_transforms.ExtResize(256),
                utils.ext_transforms.ExtToTensor(),
                utils.ext_transforms.ExtNormalize((0.5, ), (0.5, )),
            ])
            train_dataset = CamVid(self.config.dataset_path,
                                   split='train',
                                   transform=train_transforms)
            test_dataset = CamVid(self.config.dataset_path,
                                  split=self.config.test_mode,
                                  transform=test_transforms)

        # NYUv2 dataset
        if self.config.dataset == 'Nyu':
            train_transforms = utils.ext_transforms.ExtCompose([
                utils.ext_transforms.ExtResize(256),
                utils.ext_transforms.ExtRandomCrop(128, pad_if_needed=True),
                utils.ext_transforms.ExtRandomHorizontalFlip(),
                utils.ext_transforms.ExtToTensor(),
                utils.ext_transforms.ExtNormalize((0.5, ), (0.5, )),
            ])
            test_transforms = utils.ext_transforms.ExtCompose([
                utils.ext_transforms.ExtResize(256),
                utils.ext_transforms.ExtToTensor(),
                utils.ext_transforms.ExtNormalize((0.5, ), (0.5, )),
            ])
            train_dataset = NYUv2(self.config.dataset_path,
                                   split='train',
                                   transform=train_transforms)
            test_dataset = NYUv2(self.config.dataset_path,
                                  split=self.config.test_mode,
                                  transform=test_transforms)

        # Cityscapes Dataset
        elif self.config.dataset == 'Cityscapes':
            train_transforms = utils.ext_transforms.ExtCompose([
                utils.ext_transforms.ExtResize(256),
                utils.ext_transforms.ExtRandomCrop(128, pad_if_needed=True),
                utils.ext_transforms.ExtRandomHorizontalFlip(),
                utils.ext_transforms.ExtToTensor(),
                utils.ext_transforms.ExtNormalize((0.5, ), (0.5, )),
            ])
            test_transforms = utils.ext_transforms.ExtCompose([
                utils.ext_transforms.ExtResize(256),
                utils.ext_transforms.ExtToTensor(),
                utils.ext_transforms.ExtNormalize((0.5, ), (0.5, )),
            ])
            train_dataset = Cityscapes(self.config.dataset_path,
                                       split='train',
                                       mode='fine',
                                       transform=train_transforms)
            test_dataset = Cityscapes(self.config.dataset_path,
                                      split='test',
                                      mode='fine',
                                      transform=test_transforms)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=6)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            num_workers=6)

        if self.config.model == 'resnet50_pretrained':
            self.model = deeplabv3.deeplabv3_resnet50(num_classes=13,
                                                      dropout_p=0.5,
                                                      pretrained_backbone=True)
        if self.config.model == 'resnet100_pretrained':
            self.model = deeplabv3.deeplabv3_resnet101(num_classes=13,
                                                      dropout_p=0.5,
                                                      pretrained_backbone=True)
        if self.config.model == 'resnet50':
            self.model = deeplabv3.deeplabv3_resnet50(
                num_classes=13, dropout_p=0.5, pretrained_backbone=False)
        if self.config.model == 'mobilenet':
            self.model = deeplabv3.deeplabv3_mobilenet(
                num_classes=13, dropout_p=0.5, pretrained_backbone=False)

        # Uses SGD optimizer for better performance than Adam
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            momentum=self.hparams.momentum)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        # Run on GPU only. Convert model to GPU
        self.device = torch.device("cuda")
        self.model = self.model.to(self.device)

        # Learning rate scheduler
        if self.hparams.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, self.hparams.scheduler_step,
                self.hparams.scheduler_gamma)

        # Initialise a new logging directory and a tensorboard logger
        if os.path.exists(self.config.log_dir):
            raise Exception("Log directory exists")
        self.logger = SummaryWriter(log_dir=self.config.log_dir)

    def train_step(self, batch, batch_idx):
        '''
        Perform a single step (Forward and backward) of train on a given batch
        Log necessary metrics
        '''

        data, target = batch
        self.optimizer.zero_grad()
        logits = self.model(data)
        loss = utils.focal_loss(logits, target, gamma=2, ignore_index=255)
        self.train_loss += loss.item()
        loss.backward()
        self.optimizer.step()
        self.train_metrics.update(
            logits.max(1)[1].detach().cpu().numpy().astype('uint8'),
            target.detach().cpu().numpy().astype('uint8'))

    def test_step(self, batch, batch_idx):
        '''
        Perform a single forward pass on test batch
        '''
        data, target = batch
        logits = self.model(data)
        loss = utils.focal_loss(logits, target, gamma=2, ignore_index=255)
        self.test_loss += loss.item()
        self.test_metrics.update(
            logits.max(1)[1].detach().cpu().numpy().astype('uint8'),
            target.detach().cpu().numpy().astype('uint8'))

    def train_epoch(self, epoch_id):
        '''
        One epoch for training
        '''
        self.model.train()
        self.train_loss = 0
        self.train_metrics = utils.stream_metrics.StreamSegMetrics(
            n_classes=13)
        for batch_idx, batch in enumerate(
                tqdm(self.train_loader, leave=False, unit='batch',
                     ascii=True)):
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            target = target.long()
            batch = (data, target)
            self.train_step(batch, batch_idx)

        # Step the scheduler
        self.scheduler.step()

        # Logging stuff
        print("Epoch: {}".format(epoch_id))
        print("Avg Train Loss {}".format(self.train_loss / batch_idx))
        result = self.train_metrics.get_results()
        print("Pix Acc {}, mIoU {}".format(result['Overall Acc'],
                                           result['Mean IoU']))
        self.logger.add_scalar('Loss/Train', self.train_loss / batch_idx,
                               epoch_id)
        self.logger.add_scalar('Accuracy/Train', result['Overall Acc'],
                               epoch_id)
        self.logger.add_scalar('mIoU/Train', result['Mean IoU'], epoch_id)

    def test_epoch(self, epoch_id):
        '''
        Single test epoch
        '''
        self.model.eval()
        self.test_loss = 0
        self.test_metrics = utils.stream_metrics.StreamSegMetrics(n_classes=13)
        for batch_idx, batch in enumerate(self.test_loader):
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            target = target.long()
            batch = (data, target)
            self.test_step(batch, batch_idx)

        print("Avg Test Loss {}".format(self.test_loss / batch_idx))
        result = self.test_metrics.get_results()
        print("Pix Acc {}, mIoU {}".format(result['Overall Acc'],
                                           result['Mean IoU']))
        self.logger.add_scalar('Loss/Test', self.test_loss / batch_idx,
                               epoch_id)
        self.logger.add_scalar('Accuracy/Test', result['Overall Acc'],
                               epoch_id)
        self.logger.add_scalar('mIoU/Test', result['Mean IoU'], epoch_id)
        return result

    def log_qualitative(self, epoch):

        batch = next(iter(self.test_loader))
        data, target = batch
        data = data.to(self.device)
        target = target.cpu()
        logits = self.model(data).detach().cpu()
        pred = logits.max(axis=1)[1]
        pred_segmap = self.test_loader.dataset.decode_target(pred.numpy())
        pred_segmap = torch.Tensor(pred_segmap).permute(0, 3, 1, 2)
        target_segmap = self.test_loader.dataset.decode_target(target.numpy())
        target_segmap = torch.Tensor(target_segmap).permute(0, 3, 1, 2)
        data = (data.cpu()+1)/2
        grid = torch.cat([data[:8], target_segmap[:8], pred_segmap[:8]])
        grid = torchvision.utils.make_grid(grid, nrow=8)
        self.logger.add_image('Predictions', grid, epoch)

    def fit(self):
        self.best_miou = 0
        for epoch_id in range(1, self.hparams.num_epochs + 1):
            self.train_epoch(epoch_id)
            result = self.test_epoch(epoch_id)
            if result['Mean IoU'] > self.best_miou:
                self.best_miou = result['Mean IoU']
                self.logger.add_scalar('Best_mIoU', self.best_miou, epoch_id)
                self.log_qualitative(epoch_id)
                if self.config.save_checkpoint == 'best':
                    checkpoint_dict = dict(state_dict=self.model.state_dict(),
                                           hparams=vars(self.hparams),
                                           epoch=epoch_id,
                                           mIoU=self.best_miou)
                    checkpoint_path = os.path.join(self.config.log_dir,
                                                   'best.tar')
                    print("Saving best checkpoint.")
                    torch.save(checkpoint_dict, checkpoint_path)
            print('-' * 10)

        self.logger.add_hparams(vars(self.hparams), {"mIoU": self.best_miou})
        print("Best mean IoU {}".format(self.best_miou))

    def load_from_checkpint(self, path):

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("Model loaded succesfully")
