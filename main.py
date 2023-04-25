import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from generator import Generator
from discriminator import Discriminator
from utils import plot_losses, view_samples, save_train_data, generate_z
from load_data import load_data


# Global Variables
nchannels = 3
height = 64
width = 64 
latent_size = 100

batch_size = 64   # Number of real images fed to the descriminator per training cycle
num_epochs = 100


# Runner code!
def main():

    G = Generator(latent_size, nchannels)
    D = Discriminator(nchannels, height, width)

    d_optimizer = optim.Adam(D.parameters())
    g_optimizer = optim.Adam(G.parameters())

    losses, samples = train(D, G, d_optimizer, g_optimizer, latent_size)

    # Save, visualize losses and samples
    losses_fname = 'train_losses.npy'
    samples_fname = 'train_samples.pkl'
    save_train_data(losses, samples, losses_fname, samples_fname)
    plot_losses(losses_fname)
    view_samples(samples_fname)


def train(D, G, d_optimizer, g_optimizer, latent_size):
    print('Starting train!')

    samples = []
    losses = np.zeros((num_epochs, 2))

    # Constant z to track generator change
    constant_z = generate_z(4, latent_size)

    # Train time
    D.train()
    G.train()
    
    for epoch in tqdm(range(num_epochs), desc='Training Models'):
        
        # load data in batches of size batch_size. Weights are updated after predictions are made for every batch
        
        real_data_loader = load_data(dataset_type='train', batch_size=64)
        batch_num = 0
        for real_images, _ in real_data_loader:
            
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


            batch_num += 1 # Increment batch number

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

    print('Finished train')
    return losses, samples


if __name__ == '__main__':
    main()