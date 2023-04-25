import torch
import torch.optim as optim
from torch.backends.mps import is_available, is_built
import numpy as np
from tqdm import tqdm
import os

from generator import Generator
from discriminator import Discriminator
from utils import plot_losses, view_samples, save_train_data, generate_z
from load_data import load_data


# Global Variables
nchannels = 3
img_size = 64 # height and width of images
latent_size = 100

batch_size = 64 # Number of real images fed to the descriminator per training cycle
num_epochs = 100

cur_dir = os.getcwd()
data_path = os.path.join(cur_dir, 'dogs-vs-cats')

# Runner code!
def main():

    device = setup_gpu('mac')

    G = Generator(latent_size, nchannels, device).to(device)
    D = Discriminator(nchannels, device).to(device)

    d_optimizer = optim.Adam(D.parameters())
    g_optimizer = optim.Adam(G.parameters())

    losses, samples = train(D, G, d_optimizer, g_optimizer, latent_size, device)

    # Save results
    losses_fname = 'train_losses.npy'
    samples_fname = 'train_samples.pkl'
    save_train_data(losses, samples, losses_fname, samples_fname)

    # Visualize results
    # plot_losses(losses_fname)
    # view_samples(samples_fname)


def train(D, G, d_optimizer, g_optimizer, latent_size, device):
    print('Starting train!')

    samples = []
    losses = np.zeros((num_epochs, 2))

    # Constant z to track generator change
    constant_z = generate_z(4, latent_size, device)

    # Train time
    D.train()
    G.train()
    
    for epoch in tqdm(range(num_epochs), desc='Training Models'):
        
        # load data in batches of size batch_size. Weights are updated after predictions are made for every batch
        
        real_data_loader = load_data(data_path=data_path, batch_size=batch_size, img_size=img_size)
        num_batches = 37500//batch_size
        for batch_num, (real_images, _) in tqdm(enumerate(real_data_loader), desc=f'Epoch {epoch}', total=num_batches):
            real_images = real_images.to(device)
            
            # ============================================
            #            TRAIN DISCRIMINATOR
            # ============================================
            d_optimizer.zero_grad()
            
            # Get predictions of real images
            preds_real = D(real_images)
            
            # Get predictions of fake images
            z = generate_z(batch_size, latent_size, device)
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
            z = generate_z(batch_size, latent_size, device)
            fake_images = G(z)
            
            # Compute loss
            preds_fake = D(fake_images)
            g_loss = G.loss(preds_fake) 
            
            # Update generator model
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if batch_num % 50 == 0:
                message = 'Epoch [{:5d}/{:5d}] | Batch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, num_epochs, batch_num+1, num_batches, d_loss.item(), g_loss.item())
                tqdm.write(message)

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


# Setup GPU
def setup_gpu(os_type):
    available = is_available()
    built = is_built()

    if available and built:
        print('Using GPU: True')
        if os_type == 'mac':
            device = torch.device('mps')
        elif os_type == 'nvidia':
            device = torch.device('cuda')
    else:
        print('Using GPU: False')
        device = None
    
    return device
        


if __name__ == '__main__':
    main()