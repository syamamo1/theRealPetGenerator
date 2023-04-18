import torch.nn as nn
import torch.nn.functional as F


# Takes in a 1D latent vector (noise)
# Outputs a real-lookin' photo of a cat/dog
class Generator(nn.Module):
    def __init__(self, latent_size, nchannels):
        super(Generator, self).__init__()
        
        # How do I determine the output size of this model?
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 32, 3),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, 3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, 3),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nchannels, 3)
        )

    def forward(self, x):
        return self.model(x)