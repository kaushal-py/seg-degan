import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import code.network.dcgan_model as dcgan
import code.network.alexnet as alexnet
import code.network.resnet as resnet
import code.utils as utils


class DeGanSystem:
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

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])

        if self.config.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(root=self.config.dataset_path,
                                                         train=True,
                                                         transform=train_transform,
                                                         download=True)
            test_dataset = torchvision.datasets.CIFAR10(root=self.config.dataset_path,
                                                        train=False,
                                                        transform=test_transform,
                                                        download=True)
        elif self.config.dataset == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100(root=self.config.dataset_path,
                                                         train=True,
                                                         transform=train_transform,
                                                         download=True)
            test_dataset = torchvision.datasets.CIFAR100(root=self.config.dataset_path,
                                                        train=False,
                                                        transform=test_transform,
                                                        download=True)


        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=6,
            drop_last=True,
            )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=6)

        self.device = torch.device("cuda")
        model_checkpoint = torch.load(self.hparams.model_checkpoint)
        self.model.load_state_dict(model_checkpoint['state_dict'])
        print("Teacher checkpoint loaded succesfully.")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.G = dcgan.Generator(ngpu=1, nz=self.hparams.nz)
        self.D = dcgan.Discriminator(ngpu=1)
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)
        self.G.apply(self.weights_init)
        self.D.apply(self.weights_init)

        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(64,
                                       self.hparams.nz,
                                       1,
                                       1,
                                       device=self.device)
        self.real_batch = torch.full((self.hparams.batch_size, ),
                                     1,
                                     device=self.device)
        self.fake_batch = torch.full((self.hparams.batch_size, ),
                                     0,
                                     device=self.device)
        self.diversity_gt = torch.ones(
            self.config.num_classes,
            device=self.device) / self.config.num_classes

        # Used classes of CIFAR100 (Background classes used here)
        self.inc_classes = [68, 23, 33, 49, 60, 71]
        # Exclude classes from vehicles1 and vehicles2
        self.exclude_classes = [8, 13, 48, 41, 90, 58, 69, 81, 85, 89]

        self.optimizerD = torch.optim.Adam(self.D.parameters(),
                                           lr=self.hparams.lr,
                                           betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.G.parameters(),
                                           lr=self.hparams.lr,
                                           betas=(0.5, 0.999))

        if os.path.exists(self.config.log_dir):
            raise Exception("Log directory exists")
        self.logger = SummaryWriter(log_dir=self.config.log_dir)

    def train_step(self, batch, batch_idx):

        r_data, target = batch
        batch_size = r_data.shape[0]

        ## Train Discriminator ##
        self.D.zero_grad()
        r_output = self.D(r_data).view(-1)
        errD_real = self.criterion(r_output, self.real_batch[:batch_size])
        errD_real.backward()
        self.D_x += r_output.mean().item()

        noise = torch.randn(batch_size,
                            self.hparams.nz,
                            1,
                            1,
                            device=self.device)
        f_data = self.G(noise)
        f_output = self.D(f_data.detach()).view(-1)
        errD_fake = self.criterion(f_output, self.fake_batch[:batch_size])
        errD_fake.backward()
        self.D_G_z += f_output.mean().item()
        self.disc_loss += errD_real.item() + errD_fake.item()
        self.optimizerD.step()

        ## Train Generator ##
        self.G.zero_grad()
        f_output = self.D(f_data).view(-1)
        errG = self.criterion(f_output, self.real_batch[:batch_size])
        self.gen_loss += errG.item()

        ## DeGAN Losses ##
        with torch.no_grad():
            c_output = self.model(f_data).detach()
        c_softmax = F.softmax(c_output, dim=1)

        ## Entropy Loss ##
        # Calculate entropy for each pixel
        entropy_F = -1 * torch.sum(c_softmax * torch.log(c_softmax + 1e-5),
                                   dim=1)
        # Get mean accross spatial dimensions
        # entropy_y = torch.mean(torch.mean(entropy_F, dim=1), dim=1)
        # Entropy loss
        entropy_loss = torch.mean(entropy_F)

        ## Diversity Loss ##
        if self.hparams.diversity == 'entropy':

            # Get mean across spatial dimensions
            # c_softmax_pixel_mean = torch.mean(torch.mean(c_softmax, dim=2),
            #                                   dim=2)
            # c_softmax_pixel_mean is BxC

            # batch mean is C-dimensional vector
            batch_mean = torch.mean(c_softmax, dim=0)

            # Sanity check: Batch mean should add up to 1
            # print(torch.sum(batch_mean).item())

            # Diversity loss
            diversity_loss = torch.sum(
                batch_mean *
                torch.log(batch_mean))  # Maximize entropy across batch

        ## Diversity Regression Loss ##
        if self.hparams.diversity == 'regression':

            # Get mean across spatial dimensions
            # c_softmax_pixel_mean = torch.mean(torch.mean(c_softmax, dim=2),
            #                                   dim=2)
            # c_softmax_pixel_mean is BxC

            # batch mean is C-dimensional vector
            batch_mean = torch.mean(c_softmax, dim=0)

            l2_distance = (self.diversity_gt - batch_mean)**2
            weighted_l2_distance = torch.div(l2_distance, batch_mean)

            diversity_loss = weighted_l2_distance.mean()

        self.div_loss += diversity_loss.item()
        self.ent_loss += entropy_loss.item()
        total_loss = errG + self.hparams.diversity_weight * diversity_loss + self.hparams.entropy_weight * entropy_loss
        total_loss.backward()
        self.optimizerG.step()

    def train_epoch(self, epoch_id):
        self.D_x = 0
        self.D_G_z = 0
        self.disc_loss = 0
        self.gen_loss = 0
        self.ent_loss = 0
        self.div_loss = 0
        self.G.train()
        for batch_idx, batch in enumerate(
                tqdm(self.train_loader, leave=False, unit='batch',
                     ascii=True)):
            data, target = batch

            if self.config.dataset == 'cifar100':
                # data = torch.from_numpy(data.numpy()[np.isin(target, self.inc_classes)])
                data = torch.from_numpy(data.numpy()[~np.isin(target, self.exclude_classes)])

            data, target = data.to(self.device), target.to(self.device)
            batch = (data, target)
            self.train_step(batch, batch_idx)

        print("Epoch: {}".format(epoch_id))
        self.logger.add_scalar('D_x', self.D_x / batch_idx, epoch_id)
        self.logger.add_scalar('D_G_z', self.D_G_z / batch_idx, epoch_id)
        self.logger.add_scalar('Discriminator_loss',
                               self.disc_loss / batch_idx, epoch_id)
        self.logger.add_scalar('Generator_loss', self.gen_loss / batch_idx,
                               epoch_id)
        self.logger.add_scalar('Entropy_loss', self.ent_loss / batch_idx,
                               epoch_id)
        self.logger.add_scalar('Diversity_loss', self.div_loss / batch_idx,
                               epoch_id)

    def test_epoch(self, epoch_id):
        self.G.eval()
        with torch.no_grad():
            f_data = self.G(self.fixed_noise).detach()
            logits = self.model(f_data).detach().cpu()
        pred = logits.max(axis=1)[1]
        f_data = (f_data.cpu() + 1) / 2
        grid = torchvision.utils.make_grid(f_data, nrow=8)
        self.logger.add_image('Generated_images', grid, epoch_id)
        self.logger.add_histogram('Class_Distribution', pred, epoch_id)

    def fit(self):
        for epoch_id in range(1, self.hparams.num_epochs + 1):
            self.train_epoch(epoch_id)
            if epoch_id % self.config.checkpoint_interval == 0:
                self.test_epoch(epoch_id)
                checkpoint_dict = dict(
                    g_state_dict=self.G.state_dict(),
                    # d_state_dict = self.D.state_dict(),
                    hparams=vars(self.hparams),
                    epoch=epoch_id,
                )
                checkpoint_path = os.path.join(self.config.log_dir,
                                               'epoch_{}.tar'.format(epoch_id))
                print("Saving checkpoint.")
                torch.save(checkpoint_dict, checkpoint_path)

        self.logger.add_hparams(vars(self.hparams), {})

    def load_from_checkpint(self, path):

        checkpoint = torch.load(path)
        self.G.load_state_dict(checkpoint['g_state_dict'])
        print("Model loaded succesfully")

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
