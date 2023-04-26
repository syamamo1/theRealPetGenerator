import torch.nn as nn
from torch import ones

# Takes in a 1D latent vector (noise)
# Outputs a real-lookin' photo of a cat/dog
class Generator(nn.Module):
    def __init__(self, latent_size, nchannels, device):
        super(Generator, self).__init__()

        # Binary cross entropy: evaluates probabilities
        self.criterion = nn.BCELoss() 

        # GPU device!
        self.device = device
        
        # Input is size (nsamples, latent_size, 1, 1)
        # output_height = (input_height - 1) * stride - 2 * padding + kernel_size
        # output_width = (input_width - 1) * stride - 2 * padding + kernel_size
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, nchannels, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        # Setup for multiple GPUs
        # self.model = nn.DataParallel(self.model)


    # Trying to fool discriminator with generated images
    # so labels are 1
    # smoothing if we want to "generalize more"
    def loss(self, preds, smooth=False):
        nsamples = preds.size(0)

        # smooth, real labels = 0.9
        if smooth:
            labels = ones(nsamples, device = self.device)*0.9
        else:
            labels = ones(nsamples, device = self.device)

        loss = self.criterion(preds.squeeze(), labels)
        return loss


    # Forward pass
    def forward(self, x):
        return self.model(x)
