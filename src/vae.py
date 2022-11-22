import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class VAE(nn.Module):

    def __init__(self, indim, zDim=8):
        super(VAE, self).__init__()
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        channels, width, _ = indim
        self.indim = indim

        self.encoder_convolutions = nn.ParameterDict()
        self.decoder_convolutions = nn.ParameterDict()

        self.widths = {}
        self.widths[0] = width

        self.channels = {}
        self.channels[0] = channels

        self.power  = int(np.log(width-1)/np.log(2))

        for p in range(self.power):
            self.widths[p+1] = (self.widths[p] - 5 + 2*2)//2 +1 
            self.channels[p+1] = (channels*2**(p+1))
            self.encoder_convolutions[str(p+1)] = \
                nn.Conv2d(in_channels = channels*2**(p),
                          out_channels = channels*2**(p+1),
                          kernel_size = 5,
                          stride = 2,
                          padding = 2,
                          padding_mode ='reflect',
                          dtype=float)
            torch.nn.init.xavier_normal_(self.encoder_convolutions[str(p+1)].weight)

            self.decoder_convolutions[str(self.power-p)]=\
                nn.ConvTranspose2d(in_channels = channels*2**(p+1),
                                   out_channels = channels*2**(p),
                                   kernel_size=5,
                                   stride = 2,
                                   padding = 2,
                                   padding_mode ='zeros',
                                   dtype=float)
            torch.nn.init.xavier_normal_(self.decoder_convolutions[str(self.power-p)].weight)

                
        self.outdim = self.channels[self.power], self.widths[self.power], self.widths[self.power]
        self.outdim_flat = self.outdim[0]*self.outdim[1]**2

        self.encFC1 = nn.Linear(self.outdim_flat, self.outdim_flat//2, dtype=float)
        torch.nn.init.xavier_normal_(self.encFC1 .weight)

        self.encFC2mu = nn.Linear(self.outdim_flat//2, zDim, dtype=float)
        torch.nn.init.xavier_normal_(self.encFC2mu .weight)

        self.encFC2var = nn.Linear(self.outdim_flat//2, zDim, dtype=float)
        torch.nn.init.xavier_normal_(self.encFC2var .weight)

        self.decFC1 = nn.Linear(zDim, self.outdim_flat//2, dtype=float)
        torch.nn.init.xavier_normal_(self.decFC1 .weight)

        self.decFC2 = nn.Linear(self.outdim_flat//2, self.outdim_flat, dtype=float)
        torch.nn.init.xavier_normal_(self.decFC2 .weight)


    def encoder(self, x):
        # Convolutions
        for encoder in range(self.power):
            x = F.elu(self.encoder_convolutions[str(encoder+1)](x))
        # Flatten
        x = x.view(-1, self.outdim_flat)
        # Fully connected
        x = F.elu(self.encFC1(x))
        mu = self.encFC2mu(x)
        logVar = self.encFC2var(x)

        return mu, logVar

    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2 + 1e-10)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        x = F.elu(self.decFC1(z))
        x = F.elu(self.decFC2(x))
        x = x.view(-1,*self.outdim)
        for decoder in range(self.power-1):
            channels = self.channels[self.power-decoder-1]
            width = self.widths[self.power-decoder-1]
            x = F.elu(self.decoder_convolutions[str(decoder+1)](x, output_size = (-1, channels, width, width)))
        x = torch.sigmoid(self.decoder_convolutions[str(self.power)](x, output_size = (-1, *self.indim)))
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar