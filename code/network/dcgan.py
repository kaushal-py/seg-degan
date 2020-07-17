import torch
from torch import nn

class DcGanGenerator(nn.Module):
    def __init__(self, nz, nc=3, ngf=64):
        super(DcGanGenerator, self).__init__()
        self.main = nn.Sequential(

            # Uncomment for 512x512 input
            #  nn.ConvTranspose2d( nz, ngf * 64, 4, 1, 0, bias=False),
            #  nn.BatchNorm2d(ngf * 64),
            #  nn.ReLU(True),

            # Uncomment for 256x256 input
            # nn.ConvTranspose2d( nz, ngf * 32, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 32),
            # nn.ReLU(True),

            # Uncomment for 512x512 input
            # nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 32),
            # nn.ReLU(True),

            # Uncomment for 256x256 input
            # nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 16),
            # nn.ReLU(True),

            # Uncomment for 128x128 input
            nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            # This is for >= 128x128 (remove it for 64x64)
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # Uncomment for 64x64 input and 32x32 input
            # nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Uncomment for 32x32 input
            # nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),

            # Uncomment for 64x64 and 128x128 input
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)



class DcGanDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(DcGanDiscriminator, self).__init__()

        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Uncomment for >= 128x128 input
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # Uncomment for 256x256 input
            # nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 32),
            # nn.LeakyReLU(0.2, inplace=True),

            # Uncomment for 512x512 input
            # nn.Conv2d(ndf * 32, ndf * 64, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 64),
            # nn.LeakyReLU(0.2, inplace=True),

            # Uncomment for 256x256 input
            # nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),

            # Uncomment for 512x512 input
            # nn.Conv2d(ndf * 64, 1, 4, 1, 0, bias=False),

            # Uncomment for 128x128 input
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),

            # Uncomment for 32x32 input
            # nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),

            # Uncomment for 64x64 input
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
