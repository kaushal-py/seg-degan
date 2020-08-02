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
# import resnet


class Cifar10System:
    def __init__(self, config, hparams):

        self.config = config
        self.hparams = hparams
        np.random.seed(42)
        torch.manual_seed(42)

        if self.config.model == 'alexnet':
            self.model = alexnet.AlexNet(num_classes=10)
        if self.config.model == 'alexnet_half':
            self.model = alexnet.AlexNet_half(num_classes=10)
        if self.config.model == 'resnet34':
            self.model = resnet.ResNet34(num_classes=10)
        if self.config.model == 'resnet18':
            self.model = resnet.ResNet18(num_classes=10)

        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=self.config.dataset_path,
                                                     train=True,
                                                     transform=train_transform,
                                                     download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=self.config.dataset_path,
                                                    train=False,
                                                    transform=test_transform,
                                                    download=True)

        # train_len = len(train_dataset)
        # indices = list(range(train_len))
        # np.random.shuffle(indices)
        # split = int(self.hparams.val_split * train_len)
        # val_idx, train_idx = indices[:split], indices[split:]
        # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        # val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=8,
            # sampler=train_sampler
        )
        # self.val_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=self.hparams.batch_size,
        #     pin_memory=True,
        #     num_workers=8,
        #     sampler=val_sampler)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8)

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

    def train_step(self, batch, batch_idx):

        data, target = batch
        self.optimizer.zero_grad()
        logits = self.model(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        pred = logits.max(axis=1)[1]
        self.correct += torch.sum(pred == target).item()
        self.total += data.shape[0]
        self.train_loss += loss.item()
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

    def train_epoch(self, epoch_id):

        self.model.train()
        self.correct = 0
        self.total = 0
        self.train_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            batch = (data, target)
            self.train_step(batch, batch_idx)
        self.logger.add_scalar('Loss/Train', self.train_loss/batch_idx, epoch_id)
        self.logger.add_scalar('Accuracy/Train', self.correct/self.total, epoch_id)
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
        self.logger.add_scalar('Loss/'+split, self.test_loss/batch_idx, epoch_id)
        self.logger.add_scalar('Accuracy/'+split, self.correct/self.total, epoch_id)
        # return valdiation accuracy
        return self.correct/self.total

    def fit(self):

        self.best_val_acc = 0
        for epoch_id in range(1, self.hparams.epochs+1):
            self.train_epoch(epoch_id)
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
        # self.model = torch.load(path)
        print("Model loaded succesfully")
