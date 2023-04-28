import torch.nn as nn
from torch import ones

# Takes in a 1D latent vector (noise)
# Outputs a real-lookin' photo of a cat/dog
class Generator(nn.Module):
    def __init__(self, config, device=None):
        super(Generator, self).__init__()

        # yaml
        self.config = config

        # Binary cross entropy: evaluates probabilities
        self.criterion = nn.BCELoss() 

        # GPU device!
        self.device = device
        
        # Input is size (nsamples, latent_size, 1, 1)
        # ConvTranspose2d NOTE: output_size = (input_size - 1) * stride - 2 * padding + kernel_size

        # Outputs size (nsamples, nchannels, 64, 64)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(config.latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, config.nchannels, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

        # Setup for multiple GPUs
        # self.model = nn.DataParallel(self.model)


    # Trying to fool discriminator with generated images
    # so labels are 1
    # smoothing if we want to "generalize more"
    def loss(self, preds):
        nsamples = preds.size(0)

        labels = ones(nsamples, device = self.device) * self.config.real_labels

        loss = self.criterion(preds.squeeze(), labels)
        return loss


    # Forward pass
    def forward(self, x):
        return self.model(x)
