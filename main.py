import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

import pickle as pkl

from generator import Generator
from discriminator import Discriminator
from utils import real_loss, fake_loss, plot_losses
from load_data import load_data

# Runner code!
def main():
    nchannels = 1
    height = 64
    width = 64 
    latent_size = 100

    G = Generator(latent_size, nchannels)
    D = Discriminator(nchannels, height, width)

    d_optimizer = optim.Adam(D.parameters())
    g_optimizer = optim.Adam(G.parameters())

    # Binary cross entropy: evaluates probabilities
    criterion = nn.BCELoss() 

    data = load_data()
    losses = train(D, G, d_optimizer, g_optimizer, data, criterion, latent_size)
    plot_losses(losses)


# Need to finish this still!
def train(D, G, d_optimizer, g_optimizer, data, criterion, latent_size):
    num_epochs = 100

    samples = []
    losses = np.zeros((num_epochs, 2))

    # Constant z to track generator change
    sample_size = 4
    constant_z = np.random.uniform(-1, 1, size=(sample_size, latent_size))
    constant_z = torch.from_numpy(constant_z).float()

    # Train time
    D.train()
    G.train()
    for epoch in range(num_epochs):
        
        # This will certainly be different - data loader?
        for batch_num, (real_images, _) in enumerate(data):
            batch_size = real_images.size(0)
            
            # ============================================
            #            TRAIN DISCRIMINATOR
            # ============================================
            d_optimizer.zero_grad()
            
            # Get predictions of real images
            preds_real = D(real_images)
            
            # Get predictions of fake images
            z = np.random.uniform(-1, 1, size=(batch_size, latent_size))
            z = torch.from_numpy(z).float()
            fake_images = G(z)
            preds_fake = D(fake_images)
            
            # Compute loss
            d_loss_real = real_loss(preds_real, criterion, smooth=False)
            d_loss_fake = fake_loss(preds_fake, criterion)
            d_loss = d_loss_real + d_loss_fake

            # Update discriminator model
            d_loss.backward()
            d_optimizer.step()
            
            # =========================================
            #            TRAIN GENERATOR
            # =========================================
            g_optimizer.zero_grad()
            
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, latent_size))
            z = torch.from_numpy(z).float()
            fake_images = G(z)
            
            # Compute loss
            preds_fake = D(fake_images)
            g_loss = real_loss(preds_fake, criterion) # use real loss to flip labels
            
            # Update generator model
            g_loss.backward()
            g_optimizer.step()


            # Print some loss stats
            if batch_num % 400 == 0:
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, num_epochs, d_loss.item(), g_loss.item()))

        
        # Save epoch loss
        losses[epoch, 0] += g_loss.item()
        losses[epoch, 1] += d_loss.item()
         
        # Generate and save images
        G.eval() 
        generated_images = G(constant_z)
        samples.append(generated_images)
        G.train()


    # Save generated samples to .pkl
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return losses