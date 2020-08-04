import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import code.network.alexnet as alexnet
import code.network.resnet as resnet
import code.network.dcgan_model as dcgan_model
import code.utils.utils as utils

class DatafreeKDSystem:
    def __init__(self, config, hparams):

        self.config = config
        self.hparams = hparams
        np.random.seed(42)
        torch.manual_seed(42)

        if self.config.teacher == 'alexnet':
            self.teacher = alexnet.AlexNet(num_classes=10)
        if self.config.teacher == 'alexnet_half':
            self.teacher = alexnet.AlexNet_half(num_classes=10)
        if self.config.teacher == 'resnet34':
            self.teacher = resnet.ResNet34(num_classes=10)
        if self.config.teacher == 'resnet18':
            self.teacher = resnet.ResNet18(num_classes=10)

        teacher_checkpoint = torch.load(self.hparams.teacher_checkpoint)
        self.teacher.load_state_dict(teacher_checkpoint['state_dict'])
        self.teacher.eval()

        if self.config.model == 'alexnet':
            self.model = alexnet.AlexNet(num_classes=10)
        if self.config.model == 'alexnet_half':
            self.model = alexnet.AlexNet_half(num_classes=10)
        if self.config.model == 'resnet34':
            self.model = resnet.ResNet34(num_classes=10)
        if self.config.model == 'resnet18':
            self.model = resnet.ResNet18(num_classes=10)

        self.G = dcgan_model.Generator(ngpu=1, nz=self.hparams.nz)
        generator_checkpoint = torch.load(self.hparams.generator_checkpoint)
        # self.G.load_state_dict(generator_checkpoint['g_state_dict'])
        self.G.load_state_dict(generator_checkpoint)
        self.G.eval()

        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)
        self.teacher = self.teacher.to(self.device)
        self.G = self.G.to(self.device)

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )), ])
        train_dataset = torchvision.datasets.CIFAR10(root=self.config.dataset_path,
                                                     train=True,
                                                     transform=train_transform,
                                                     download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=self.config.dataset_path,
                                                    train=False,
                                                    transform=test_transform,
                                                    download=True)


        # self.train_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=self.hparams.batch_size,
        #     pin_memory=True,
        #     num_workers=6,
        #     sampler=train_sampler)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=6)

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.hparams.lr,
                                         momentum=0.9,
                                         weight_decay=5e-4)
        # self.optimizer = torch.optim.Adam(self.model.parameters(),
        #                                  lr=self.hparams.lr,
        #                                  weight_decay=5e-4
        #                                  )

        # Learning rate scheduler
        if self.hparams.lr_scheduler == 'cyclic':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.hparams.lr,
                max_lr=self.hparams.max_lr,
                step_size_up=self.hparams.step_size_up,
                step_size_down = self.hparams.step_size_down,
            )
        elif self.hparams.lr_scheduler == 'step':
            # self.scheduler = torch.optim.lr_scheduler.StepLR(
            #         self.optimizer,
            #         step_size=self.hparams.lr_step_size,
            #         gamma=self.hparams.lr_gamma,
            # )
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=self.hparams.lr_milestones,
                    gamma=self.hparams.lr_gamma
            )

        # Initialise a new logging directory and a tensorboard logger
        if os.path.exists(self.config.log_dir):
            raise Exception("Log directory exists")
        self.logger = SummaryWriter(log_dir=self.config.log_dir)

    def train_step(self):

        with torch.no_grad():
            noise = torch.randn(self.hparams.batch_size,
                                self.hparams.nz,
                                1,
                                1,
                                device=self.device)
            data = self.G(noise).detach()
        self.optimizer.zero_grad()
        logits_s = self.model(data)
        with torch.no_grad():
            logits_t = self.teacher(data).detach()
            target = logits_t.max(axis=1)[1]
        ce_loss = utils.soft_cross_entropy(logits_s, logits_t)
        # ce_loss = F.cross_entropy(logits_s, target)
        T = self.hparams.temperature
        kd_loss = F.kl_div(F.log_softmax(logits_s/T, dim=1), F.softmax(logits_t/T, dim=1))
        loss = kd_loss * self.hparams.alpha * T * T + (1.0-self.hparams.alpha) * ce_loss
        loss.backward()
        pred = logits_s.max(axis=1)[1]
        self.correct += torch.sum(pred == target).item()
        self.total += data.shape[0]
        self.train_loss += loss.item()
        self.train_kd_loss += kd_loss.item()
        self.train_ce_loss += ce_loss.item()
        self.optimizer.step()
        # self.scheduler.step()

    def test_step(self, batch, batch_idx):

        data, target = batch
        with torch.no_grad():
            logits = self.model(data)
        loss = F.cross_entropy(logits, target)
        pred = logits.max(axis=1)[1]
        self.correct += torch.sum(pred == target).item()
        self.total += data.shape[0]
        self.test_loss += loss.item()

    def train_epoch(self, epoch_id, step=True):

        self.model.train()
        self.correct = 0
        self.total = 0
        self.train_loss = 0
        self.train_kd_loss = 0
        self.train_ce_loss = 0
        for batch_idx in range(self.hparams.batch_length):
            self.train_step()
        self.logger.add_scalar('Loss/Train', self.train_loss/batch_idx, epoch_id)
        self.logger.add_scalar('Loss/KD', self.train_kd_loss/batch_idx, epoch_id)
        self.logger.add_scalar('Loss/CE', self.train_ce_loss/batch_idx, epoch_id)
        self.logger.add_scalar('Accuracy/Train', self.correct/self.total, epoch_id)
        if step:
            self.scheduler.step()

    def test_epoch(self, epoch_id, split='Validation'):

        self.model.eval()
        self.correct = 0
        self.total = 0
        self.test_loss = 0
        if split == 'Validation':
            # loader = self.val_loader
            raise Exception("Selected validation dataloader")
        elif split == 'Test':
            loader = self.test_loader
        for batch_idx, batch in enumerate(loader):
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            batch = (data, target)
            self.test_step(batch, batch_idx)
        self.logger.add_scalar('Loss/'+split, self.train_loss/batch_idx, epoch_id)
        self.logger.add_scalar('Accuracy/'+split, self.correct/self.total, epoch_id)
        # return valdiation accuracy
        return self.correct/self.total

    def fit(self):

        self.best_val_acc = 0
        for epoch_id in range(1, self.hparams.epochs+1):
            self.train_epoch(epoch_id, step=True)
            # acc = self.test_epoch(epoch_id, split='Validation')
            test_acc = self.test_epoch(epoch_id, split='Test')
            print("Epoch {}: Test accuracy: {}".format(epoch_id, test_acc))
            # if acc > self.best_val_acc:
            #     self.best_val_acc = acc
            #     test_acc = self.test_epoch(epoch_id, split='Test')
            #     print("Best acc: ", test_acc)
            #     checkpoint_dict = dict(
            #         hparams = vars(self.hparams),
            #         config = vars(self.config),
            #         state_dict = self.model.state_dict(),
            #         optimizer_dict = self.optimizer.state_dict(),
            #         lr_dict = self.scheduler.state_dict(),
            #     )
            #     checkpoint_path = os.path.join(self.config.log_dir,
            #                                    'best.tar')
            #     print("Saving best checkpoint.")
            #     torch.save(checkpoint_dict, checkpoint_path)
        # print("Best Accuracy: ", self.best_val_acc)
        
        # for epoch_id in range(self.hparams.epochs+1, self.hparams.epochs+21):
        #     self.train_epoch(epoch_id, step=False)
        #     test_acc = self.test_epoch(epoch_id, split='Test')
        #     print("Epoch {}: Test accuracy: {}".format(epoch_id, test_acc))

        checkpoint_dict = dict(
            hparams = vars(self.hparams),
            config = vars(self.config),
            state_dict = self.model.state_dict(),
            optimizer_dict = self.optimizer.state_dict(),
            lr_dict = self.scheduler.state_dict(),
        )
        checkpoint_path = os.path.join(self.config.log_dir,
                                       'last.tar')
        print("Saving last checkpoint.")
        torch.save(checkpoint_dict, checkpoint_path)

    def load_from_checkpint(self, path):

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("Model loaded succesfully")
