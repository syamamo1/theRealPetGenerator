import torch
from tqdm import tqdm
from datetime import datetime
from discriminator import Discriminator
from generator import Generator
import numpy as np


# Print num parameters, num GPUs
def print_pytorch_stats(config):
    D = Discriminator(config)
    G = Generator(config)
    nD = round(sum(p.numel() for p in D.parameters())/1e6, 3)
    nG = round(sum(p.numel() for p in G.parameters())/1e6, 3)
    print(f'Num trainable Discriminator: {nD} M')
    print(f'Num trainable Generator: {nG} M')


# Print messages about training
def print_message(epoch, num_epochs, batch_num, num_batches, d_loss, g_loss, acc_real, acc_fake):
    message = '''
               Epoch [{:5d}/{:5d}] | Batch [{:5d}/{:5d}] 
                  acc_real: {:6.4f} | d_loss: {:6.4f}
                  acc_fake: {:6.4f} | g_loss: {:6.4f}  
                '''.format(
                        epoch, num_epochs, batch_num, num_batches, 
                        acc_real, d_loss.item(), 
                        acc_fake, g_loss.item())
    tqdm.write(message)


# Write message to file
def log_message(info1, info2):
    config, rank, epoch, batch_num, num_batches, d_loss, g_loss = info1
    acc_real, acc_fake, av_real_pred, av_fake_pred = info2

    message = '''
########################################################################################

               Epoch [{:5d}/{:5d}] | Batch [{:5d}/{:5d}] --> Rank {}
                  acc_real: {:3.2f} | d_loss: {:6.4f} | av_real_pred: {:3.2f}
                  acc_fake: {:3.2f} | g_loss: {:6.4f} | av_fake_pred: {:3.2f}
                  '''.format(
                        epoch, config.num_epochs, batch_num, num_batches, rank,
                        acc_real, d_loss.item(), av_real_pred,
                        acc_fake, g_loss.item(), av_fake_pred)
    
    
    with open(config.log_fname, 'a') as f:
        f.write(message)


# Write time to file
def log_time(config, epoch, start_time):
    cur_time = datetime.now()
    train_time = (cur_time - start_time).strftime("%H:%M:%S")
    cur_time = cur_time.strftime("%H:%M:%S")
    message = f'\nTrain time for epoch {epoch}: {train_time} @ {cur_time}\n' 

    with open(config.log_fname, 'a') as f:
        f.write(message)


# Clears file
def newfile(config):
    with open(config.log_fname, 'w') as _:
        pass
