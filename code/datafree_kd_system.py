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
import code.network.segmentation.deeplabv3 as deeplabv3
import code.network.dcgan as dcgan
import code.utils as utils

class DatafreeKDSystem:

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
            test_dataset = CamVid(self.config.dataset_path, split=self.config.test_mode, transform=test_transforms)

        elif self.config.dataset == 'Cityscapes':
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
            train_dataset = Cityscapes(self.config.dataset_path, split='train', mode='fine', transform=train_transforms)
            test_dataset = Cityscapes(self.config.dataset_path, split='val', mode='fine', transform=test_transforms)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, shuffle=True, num_workers=6)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.hparams.test_batch_size, shuffle=False, num_workers=6)


        if self.config.teacher == 'resnet50_pretrained':
            self.teacher = deeplabv3.deeplabv3_resnet50(num_classes=11, dropout_p=0.5, pretrained_backbone=True)
        if self.config.teacher == 'mobilenet_pretrained':
            self.teacher = deeplabv3.deeplabv3_mobilenet(num_classes=11, dropout_p=0.5, pretrained_backbone=True)
        if self.config.teacher == 'resnet50':
            self.teacher = deeplabv3.deeplabv3_resnet50(num_classes=11, dropout_p=0.5, pretrained_backbone=False)
        if self.config.teacher == 'mobilenet':
            self.teacher = deeplabv3.deeplabv3_mobilenet(num_classes=11, dropout_p=0.5, pretrained_backbone=False)


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
        teacher_checkpoint = torch.load(self.config.teacher_checkpoint)
        self.teacher.load_state_dict(teacher_checkpoint['state_dict'])
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()
        print("Teacher checkpoint loaded successfully.")
        generator_checkpoint = torch.load(self.config.generator_checkpoint)
        self.hparams.nz = generator_checkpoint['hparams']['nz']
        self.G = dcgan.DcGanGenerator(nz=generator_checkpoint['hparams']['nz'])
        self.G.load_state_dict(generator_checkpoint['g_state_dict'])
        self.G.to(self.device)
        self.G.eval()
        print("Generator model loaded successfully")

        if self.hparams.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.hparams.scheduler_step, self.hparams.scheduler_gamma)


        if os.path.exists(self.config.log_dir):
            raise Exception("Log directory exists")
        self.logger = SummaryWriter(log_dir=self.config.log_dir)


    def train_step(self):

        self.optimizer.zero_grad()
        noise = torch.randn(self.hparams.train_batch_size, self.hparams.nz, 1, 1, device=self.device)
        with torch.no_grad():
            data = self.G(noise).detach()
        s_logits = self.model(data)
        with torch.no_grad():
            t_logits = self.teacher(data)
        kd_loss = utils.soft_cross_entropy(s_logits, t_logits)
        loss = kd_loss
        # loss = ce_loss + self.hparams.kd_weight*kd_loss
        self.train_loss += loss.item()
        self.train_kd_loss += kd_loss.item()
        loss.backward()
        self.optimizer.step()

    def test_step(self, batch, batch_idx):

        data, target = batch
        s_logits = self.model(data)
        with torch.no_grad():
            t_logits = self.teacher(data)
        kd_loss = utils.soft_cross_entropy(s_logits, t_logits)
        loss = kd_loss
        self.test_loss += loss.item()
        self.test_kd_loss += kd_loss.item()
        self.test_metrics.update(s_logits.max(1)[1].detach().cpu().numpy().astype('uint8'), target.detach().cpu().numpy().astype('uint8'))

    def train_epoch(self, epoch_id):
        self.model.train()
        self.train_loss = 0
        self.train_kd_loss = 0
        for batch_idx in tqdm(range(10), leave=False, unit='batch', ascii=True):
            self.train_step()
        self.scheduler.step()
        print("Epoch: {}".format(epoch_id))
        print("Avg Train Loss {}".format(self.train_loss/batch_idx))
        self.logger.add_scalar('Loss/Train', self.train_loss/batch_idx, epoch_id)
        self.logger.add_scalar('KD_Loss/Train', self.train_kd_loss/batch_idx, epoch_id)

    def test_epoch(self, epoch_id):
        self.model.eval()
        self.test_loss = 0
        self.test_kd_loss = 0
        self.test_metrics = utils.stream_metrics.StreamSegMetrics(n_classes=11)
        for batch_idx, batch in enumerate(self.test_loader):
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            target = target.long()
            batch = (data, target)
            self.test_step(batch, batch_idx)
        print("Avg Test Loss {}".format(self.test_loss/batch_idx))
        result = self.test_metrics.get_results()
        print("Pix Acc {}, mIoU {}".format(result['Overall Acc'], result['Mean IoU']))
        self.logger.add_scalar('Loss/Test', self.test_loss/batch_idx, epoch_id)
        self.logger.add_scalar('KD_Loss/test', self.test_kd_loss/batch_idx, epoch_id)
        self.logger.add_scalar('Accuracy/Test', result['Overall Acc'], epoch_id)
        self.logger.add_scalar('mIoU/Test', result['Mean IoU'], epoch_id)
        return result

    def fit(self):
        self.best_miou = 0
        for epoch_id in range(1, self.hparams.num_epochs+1):
            self.train_epoch(epoch_id)
            result = self.test_epoch(epoch_id)
            if result['Mean IoU'] > self.best_miou:
                self.best_miou = result['Mean IoU']
                self.logger.add_scalar('Best_mIoU', self.best_miou, epoch_id)
                if self.config.save_checkpoint == 'best':
                    checkpoint_dict = dict(
                            state_dict = self.model.state_dict(),
                            hparams = vars(self.hparams),
                            epoch = epoch_id,
                            mIoU = self.best_miou
                            )
                    checkpoint_path = os.path.join(self.config.log_dir, 'best.tar')
                    print("Saving best checkpoint.")
                    torch.save(checkpoint_dict, checkpoint_path)
            print('-'*10)

        self.logger.add_hparams(vars(self.hparams), {"mIoU": self.best_miou})
        print("Best mean IoU {}".format(self.best_miou))


