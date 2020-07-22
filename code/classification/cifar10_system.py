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


class Cifar10System:
    def __init__(self, config, hparams):

        self.config = config
        self.hparams = hparams
        np.random.seed(1337)
        torch.manual_seed(1337)

        if self.config.model == 'alexnet':
            self.model = alexnet.AlexNet(num_classes=10)
        if self.config.model == 'alexnet_half':
            self.model = alexnet.AlexNet_half(num_classes=10)

        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=self.config.dataset_path,
                                                     train=True,
                                                     transform=train_transform,
                                                     download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=self.config.dataset_path,
                                                    train=False,
                                                    transform=test_transform,
                                                    download=True)

        train_len = len(train_dataset)
        indices = list(range(train_len))
        np.random.shuffle(indices)
        split = int(self.hparams.val_split * train_len)
        train_idx, val_idx = indices[:split], indices[split:]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=6,
            sampler=train_sampler)
        self.val_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=6,
            sampler=val_sampler)
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

        # Learning rate scheduler
        if self.hparams.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.hparams.lr,
                max_lr=self.hparams.max_lr)

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
        self.scheduler.step()

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

    def test_epoch(self, epoch_id, split='Validation'):

        self.correct = 0
        self.total = 0
        self.test_loss = 0
        if split == 'Validation':
            loader = self.val_loader
        elif split == 'Test':
            loader = self.test_loader
        for batch_idx, batch in enumerate(self.val_loader):
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
            self.train_epoch(epoch_id)
            acc = self.test_epoch(epoch_id, split='Validation')
            print("Epoch {}: Validation accuracy: {}".format(epoch_id, acc))
            if acc > self.best_val_acc:
                self.best_val_acc = acc
                test_acc = self.test_epoch(epoch_id, split='Test')
                print("Best acc: ", test_acc)
                checkpoint_dict = dict(
                    hparams = vars(self.hparams),
                    config = vars(self.config),
                    state_dict = self.model.state_dict(),
                    optimizer_dict = self.optimizer.state_dict(),
                    lr_dict = self.scheduler.state_dict(),
                )
                checkpoint_path = os.path.join(self.config.log_dir,
                                               'best.tar')
                print("Saving best checkpoint.")
                torch.save(checkpoint_dict, checkpoint_path)
        print("Best Accuracy: ", self.best_val_acc)

    def load_from_checkpint(self, path):

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("Model loaded succesfully")
