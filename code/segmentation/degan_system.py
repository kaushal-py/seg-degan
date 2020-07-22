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
import code.network.dcgan as dcgan
import code.utils as utils


class DeGanSystem:
    def __init__(self, config, hparams):

        self.config = config
        self.hparams = hparams

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
                                  split='val',
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
            num_workers=6,
            drop_last=True)
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
            self.model = deeplabv3.deeplabv3_resnet101(
                num_classes=13, dropout_p=0.5, pretrained_backbone=True)
        if self.config.model == 'resnet50':
            self.model = deeplabv3.deeplabv3_resnet50(
                num_classes=13, dropout_p=0.5, pretrained_backbone=False)
        if self.config.model == 'mobilenet':
            self.model = deeplabv3.deeplabv3_mobilenet(
                num_classes=13, dropout_p=0.5, pretrained_backbone=False)

        self.device = torch.device("cuda")
        model_checkpoint = torch.load(self.hparams.model_checkpoint)
        self.model.load_state_dict(model_checkpoint['state_dict'])
        print("Teacher checkpoint loaded succesfully.")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.G = dcgan.DcGanGenerator(self.hparams.nz)
        self.D = dcgan.DcGanDiscriminator()
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)
        self.G.apply(self.weights_init)
        self.D.apply(self.weights_init)

        self.criterion = nn.BCEWithLogitsLoss()
        self.fixed_noise = torch.randn(self.hparams.train_batch_size,
                                       self.hparams.nz,
                                       1,
                                       1,
                                       device=self.device)
        self.real_batch = torch.full((self.hparams.train_batch_size, ),
                                     1,
                                     device=self.device)
        self.fake_batch = torch.full((self.hparams.train_batch_size, ),
                                     0,
                                     device=self.device)
        self.diversity_gt = torch.ones(
            self.config.num_classes,
            device=self.device) / self.config.num_classes

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

        ## Train Discriminator ##
        self.D.zero_grad()
        r_output = self.D(r_data).view(-1)
        errD_real = self.criterion(r_output, self.real_batch)
        errD_real.backward()
        self.D_x += torch.sigmoid(r_output).mean().item()

        noise = torch.randn(self.hparams.train_batch_size,
                            self.hparams.nz,
                            1,
                            1,
                            device=self.device)
        f_data = self.G(noise)
        f_output = self.D(f_data.detach()).view(-1)
        errD_fake = self.criterion(f_output, self.fake_batch)
        errD_fake.backward()
        self.D_G_z += torch.sigmoid(f_output).mean().item()
        self.disc_loss += errD_real.item() + errD_fake.item()
        self.optimizerD.step()

        ## Train Generator ##
        self.G.zero_grad()
        f_output = self.D(f_data).view(-1)
        errG = self.criterion(f_output, self.real_batch)
        self.gen_loss += errG.item()

        ## DeGAN Losses ##
        c_output = self.model(f_data)
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
            c_softmax_pixel_mean = torch.mean(torch.mean(c_softmax, dim=2),
                                              dim=2)
            # c_softmax_pixel_mean is BxC

            # batch mean is C-dimensional vector
            batch_mean = torch.mean(c_softmax_pixel_mean, dim=0)

            # Sanity check: Batch mean should add up to 1
            # print(torch.sum(batch_mean).item())

            # Diversity loss
            diversity_loss = torch.sum(
                batch_mean *
                torch.log(batch_mean))  # Maximize entropy across batch

        ## Diversity Regression Loss ##
        if self.hparams.diversity == 'regression':

            # Get mean across spatial dimensions
            c_softmax_pixel_mean = torch.mean(torch.mean(c_softmax, dim=2),
                                              dim=2)
            # c_softmax_pixel_mean is BxC

            # batch mean is C-dimensional vector
            batch_mean = torch.mean(c_softmax_pixel_mean, dim=0)

            l2_distance = (self.diversity_gt - batch_mean)**2
            weighted_l2_distance = torch.div(l2_distance, batch_mean)

            diversity_loss = weighted_l2_distance.mean()

        self.div_loss += diversity_loss.item()
        self.ent_loss += entropy_loss.item()
        total_loss = errG + self.hparams.diversity_weight * diversity_loss + self.hparams.entropy_weight * entropy_loss
        total_loss.backward()
        self.optimizerG.step()

    def train_epoch(self, epoch_id):
        self.model.train()
        self.D_x = 0
        self.D_G_z = 0
        self.disc_loss = 0
        self.gen_loss = 0
        self.ent_loss = 0
        self.div_loss = 0
        for batch_idx, batch in enumerate(
                tqdm(self.train_loader, leave=False, unit='batch',
                     ascii=True)):
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            target = target.long()
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
        with torch.no_grad():
            f_data = self.G(self.fixed_noise)

        logits = self.model(f_data).detach().cpu()
        pred = logits.max(axis=1)[1]
        pred_segmap = self.test_loader.dataset.decode_target(pred.numpy())
        pred_segmap = torch.Tensor(pred_segmap).permute(0, 3, 1, 2)
        f_data = (f_data.cpu() + 1) / 2
        grid = torch.cat([f_data[:8], pred_segmap[:8]])
        grid = torchvision.utils.make_grid(grid, nrow=8)
        self.logger.add_image('Generated_images', grid, epoch_id)

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
