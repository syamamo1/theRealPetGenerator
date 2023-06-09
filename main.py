import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from easydict import EasyDict
import torch.multiprocessing as mp
import torch.distributed as dist
import math

from generator import Generator
from discriminator import Discriminator
from utils import setup_gpu, generate_z, generate_zeros1, generate_zeros2
from utils import save_train_data, save_models, load_models, weights_init
from visualizers import plot_losses, plot_accuracy, plot_together
from visualizers import view_samples, view_dataset, view_generated_samples
from logs import print_message, log_pytorch_stats, log_message, log_time, newfile, log
from load_data import load_data


# Must specify this
remote = 0


# Filenames
config_path = 'config.yaml'
data_path = 'dogs-vs-cats'


# Get correct directory
cwd = ''
if remote: 
    cwd = os.getcwd() 
    cwd = os.path.join(cwd, 'course', 'cs1430', 'theRealPetGenerator')
    data_path = os.path.join(cwd, data_path)
    config_path = os.path.join(cwd, config_path) 


# Load config file
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)


# Runner code!
def main():

    # Train mode
    if config.train_mode:
        log(config, config)
        device, world_size = setup_gpu(config)
        log_pytorch_stats(config)

        if config.multiple_gpus:          
            log(config, f'Spawning train() on {world_size} GPUs!')
            newfile(config)
            mp.spawn(train, args=(config, world_size), nprocs=world_size) 

        else:
            log(config, 'No spawning here!')
            train(device, config, world_size)
            

    # Visualize results
    if config.eval_mode:
        # plot_together(config) # plot losses, acc, av_av
        # view_samples(config) # view samples through epochs
        # view_generated_samples(config) # view samples generated in generate mode
        view_dataset(config, data_path) # view train data

    # Generate images
    if config.generate_mode:
        log(config, config)
        device, world_size = setup_gpu(config)
        log_pytorch_stats(config)

        if config.multiple_gpus:          
            log(config, f'Spawning generate_images() on {world_size} GPUs!')
            newfile(config)
            mp.spawn(generate_images, args=(config, world_size), nprocs=world_size) 

        else:
            log(config, 'No spawning here!')
            generate_images(device, config, world_size)


# Rank is device (torch.cuda/torch.mps) for 1GPU
# or 1...n for multi GPU
def train(rank, config, world_size):
    # Load data, models
    data_loader = load_data(config, data_path, world_size, rank)
    
    G, D = make_models(config, rank, world_size)
    # Randomly initialize weights to N(0, 0.02)
    G.apply(weights_init)
    D.apply(weights_init)

   
    # Load optimizers
    d_optimizer = optim.Adam(D.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    g_optimizer = optim.Adam(G.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    # Constant z to track generator change
    constant_z = generate_z(config, config.num_constant, rank)
    samples, losses, accuracies, av_preds = generate_zeros1(config)

    # Train time
    D.train()
    G.train()

    # Iterate epochs
    num_batches = math.ceil(len(data_loader.dataset)/(world_size*config.batch_size))
    for epoch in tqdm(np.arange(1, config.num_epochs+1), desc='Training Models'):
        if not config.multiple_gpus: log(config, f'Starting epoch {epoch}')
        
        # Iterate batches
        d_losses, g_losses, real_accs, fake_accs, realpreds, fakepreds = generate_zeros2(num_batches)
        start_time = datetime.datetime.now()
        for batch_num, (real_images, _) in enumerate(data_loader, 1):
            real_images = real_images.to(rank)
            
            # ============================================
            #            TRAIN DISCRIMINATOR
            # ============================================

            # Update discriminator less bc 
            # performing too well
            if (batch_num % config.update_every == 1) or (config.update_every == 1):
                d_optimizer.zero_grad()
                
                # Get predictions of real images
                preds_real = D(real_images)
                
                # Get predictions of fake images
                z = generate_z(config, config.batch_size, rank)
                fake_images = G(z)
                preds_fake = D(fake_images)
                
                # Compute loss
                d_loss = D.loss(preds_real, preds_fake)

                # Update discriminator model
                d_loss.backward()
                d_optimizer.step()
            
            # =========================================
            #            TRAIN GENERATOR
            # =========================================
            g_optimizer.zero_grad()
            
            # Generate fake images
            z = generate_z(config, config.batch_size, rank)
            fake_images = G(z)
            
            # Compute loss
            preds_fake = D(fake_images)
            g_loss = G.loss(preds_fake) 
            
            # Update generator model
            g_loss.backward()
            g_optimizer.step()

            # Save some stats
            av_real_pred = torch.mean(preds_real).item()
            av_fake_pred = torch.mean(preds_fake).item()
            realpreds[batch_num-1] += av_real_pred
            fakepreds[batch_num-1] += av_fake_pred

            d_losses[batch_num-1] += d_loss.item()
            g_losses[batch_num-1] += g_loss.item()

            acc_real, acc_fake = D.accuracy(preds_real, preds_fake)
            real_accs[batch_num-1] += acc_real
            fake_accs[batch_num-1] += acc_fake

            # Print some loss/pred/accuracy stats
            if batch_num % config.print_every == 0:
                if (config.multiple_gpus) and (rank == 0):
                    # dist.barrier()
                    info1 = config, rank, world_size, epoch, batch_num, num_batches, d_loss, g_loss
                    info2 = acc_real, acc_fake, av_real_pred, av_fake_pred
                    log_message(info1, info2)
                elif not config.multiple_gpus:
                    print_message(epoch, config.num_epochs, batch_num, 
                                num_batches, d_loss, g_loss, acc_real, acc_fake)


        # After each epoch.....

        # Save average prediction
        av_preds[epoch-1, 0] += np.mean(realpreds)
        av_preds[epoch-1, 1] += np.mean(fakepreds)

        # Save average accuracies
        accuracies[epoch-1, 0] += np.mean(real_accs)
        accuracies[epoch-1, 1] += np.mean(fake_accs)

        # Save epoch loss
        losses[epoch-1, 0] += np.mean(g_losses)
        losses[epoch-1, 1] += np.mean(d_losses)
         
        # Generate and save images
        G.eval() 
        with torch.no_grad():
            generated_images = G(constant_z)
            samples[epoch-1] += generated_images.detach().cpu().numpy()
        G.train()

        # Print epoch duration
        # print(torch.cuda.memory_summary())
        if config.multiple_gpus: log_time(config, epoch, start_time)
        else: log(config, f'Train time for epoch {epoch}:', datetime.datetime.now()-start_time)


    # DONE-ZO
    save_train_data(config, cwd, losses, samples, accuracies, av_preds)
    save_models(config, G, D, cwd)
    log(config, 'Finished train')



# Generate images to eval model
def generate_images(rank, config, world_size):

    # Load models
    G, D = make_models(config, rank, world_size)
    G, _ = load_models(config, G, D, cwd)

    # Generate noise for generator input
    z = generate_z(config, config.num_generate, rank)

    # Generate and save images
    G.eval() 
    with torch.no_grad():
        generated_images = G(z)
        generated_images = generated_images.detach().cpu().numpy()

    # Save images
    generated_fname = os.path.join(cwd, config.generated_fname)
    np.save(generated_fname, generated_images)
    log(config, f'Saved {config.num_generate} generated images to: {generated_fname}')



# Load Generator, Discriminator
def make_models(config, rank, world_size):
    # Load models
    if config.multiple_gpus:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        G = Generator(config, rank).to(rank)
        D = Discriminator(config, rank).to(rank)
        G.model = DDP(G.model, device_ids=[rank], broadcast_buffers=False)
        D.model = DDP(D.model, device_ids=[rank], broadcast_buffers=False)  
        log(config, f'Loaded models! Rank {rank}')

    else:
        G = Generator(config, rank).to(rank)
        D = Discriminator(config, rank).to(rank)
        log(config, 'Loaded models!')

    return G, D



if __name__ == '__main__':
    start_time = datetime.datetime.now()

    main()

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time

    log(config, f'Total time: {elapsed_time} @ {end_time}')