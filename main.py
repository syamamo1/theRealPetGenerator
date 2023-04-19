import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

from generator import Generator
from discriminator import Discriminator
from utils import plot_losses, save_train_data, generate_z
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

    data = load_data()
    losses, samples = train(D, G, d_optimizer, g_optimizer, data, latent_size)
    save_train_data(losses, samples)
    plot_losses(losses)


# Need to finish this still!
def train(D, G, d_optimizer, g_optimizer, data, latent_size):
    num_epochs = 100

    samples = []
    losses = np.zeros((num_epochs, 2))

    # Constant z to track generator change
    constant_z = generate_z(4, latent_size)

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
            z = generate_z(batch_size, latent_size)
            fake_images = G(z)
            preds_fake = D(fake_images)
            
            # Compute loss
            d_loss = D.loss(preds_real, preds_fake, smooth=False)

            # Update discriminator model
            d_loss.backward()
            d_optimizer.step()
            
            # =========================================
            #            TRAIN GENERATOR
            # =========================================
            g_optimizer.zero_grad()
            
            # Generate fake images
            z = generate_z(batch_size, latent_size)
            fake_images = G(z)
            
            # Compute loss
            preds_fake = D(fake_images)
            g_loss = G.loss(preds_fake) 
            
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

    return losses, samples