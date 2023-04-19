import torch.nn as nn

# Takes in a 1D latent vector (noise)
# Outputs a real-lookin' photo of a cat/dog
class Generator(nn.Module):
    def __init__(self, latent_size, nchannels):
        super(Generator, self).__init__()
        
        # output_height = (input_height - 1) * stride - 2 * padding + kernel_size
        # output_width = (input_width - 1) * stride - 2 * padding + kernel_size
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 32, kernel_size=4, stride=4, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 128, 4, 4, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nchannels, 4, 4, 0)
        )

    def forward(self, x):
        return self.model(x)