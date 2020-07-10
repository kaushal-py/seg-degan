import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from code.dataset.camvid import CamVid
import code.network.segmentation.deeplabv3 as deeplabv3
import code.utils as utils

class SegmentationSystem:


    def __init__(self, config, hparams):

        self.config = config
        self.hparams = hparams

        if self.config.dataset == 'CamVid':
            train_transforms = utils.ext_transforms.ExtCompose([
                utils.ext_transforms.ExtResize(256),
                utils.ext_transforms.ExtRandomCrop(128, pad_if_needed=True),
                utils.ext_transforms.ExtRandomHorizontalFlip(),
                utils.ext_transforms.ExtToTensor(),
                utils.ext_transforms.ExtNormalize((0.5,), (0.5,)),
            ])
            test_transforms = utils.ext_transforms.ExtCompose([
                utils.ext_transforms.ExtResize(256),
                utils.ext_transforms.ExtToTensor(),
                utils.ext_transforms.ExtNormalize((0.5,), (0.5,)),
            ])
            train_dataset = CamVid(self.config.dataset_path, split='train', transform=train_transforms)
            test_dataset = CamVid(self.config.dataset_path, split='test', transform=test_transforms)


        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, shuffle=True, num_workers=6)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.hparams.test_batch_size, shuffle=False, num_workers=6)


        if self.config.model == 'resnet50_pretrained':
            self.model = deeplabv3.deeplabv3_resnet50(num_classes=11, dropout_p=0.5, pretrained_backbone=True)
        if self.config.model == 'mobilenet_pretrained':
            self.model = deeplabv3.deeplabv3_mobilenet(num_classes=11, dropout_p=0.5, pretrained_backbone=True)
        if self.config.model == 'resnet50':
            self.model = deeplabv3.deeplabv3_resnet50(num_classes=11, dropout_p=0.5, pretrained_backbone=False)
        if self.config.model == 'mobilenet':
            self.model = deeplabv3.deeplabv3_mobilenet(num_classes=11, dropout_p=0.5, pretrained_backbone=False)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
        self.device = torch.device("cuda")
        self.model = self.model.to(self.device)

        if self.hparams.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.hparams.scheduler_step, self.hparams.scheduler_gamma)


    def train_step(self, batch, batch_idx):

        data, target = batch
        print(data.shape, target.shape)
        self.optimizer.zero_grad()
        logits = self.model(data)
        loss = utils.focal_loss(logits, target, gamma=2, ignore_index=255)
        self.train_loss += loss.item()
        loss.backward()
        self.optimizer.step()

    def test_step(self, batch, batch_idx):

        data, target = batch
        print(data.shape, target.shape)
        logits = self.model(data)
        loss = utils.focal_loss(logits, target, gamma=2, ignore_index=255)
        self.test_loss += loss.item()

    def train_epoch(self, epoch_id):
        self.model.train()
        self.train_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            target = target.long()
            batch = (data, target)
            self.train_step(batch, batch_idx)
        self.scheduler.step()
        print("Avg Train Loss {}".format(self.train_loss/batch_idx))

    def test_epoch(self, epoch_id):
        self.model.eval()
        self.test_loss = 0
        for batch_idx, batch in enumerate(self.test_loader):
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            target = target.long()
            batch = (data, target)
            self.test_step(batch, batch_idx)
        print("Avg Test Loss {}".format(self.test_loss/batch_idx))

    def fit(self):
        for epoch_id in range(1, self.hparams.num_epochs+1):
            self.train_epoch(epoch_id)
            self.test_epoch(epoch_id)


