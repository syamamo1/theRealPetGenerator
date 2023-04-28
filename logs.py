import torch
from tqdm import tqdm
import datetime
from discriminator import Discriminator
from generator import Generator


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
def log_message(config, rank, epoch, batch_num, num_batches, d_loss, g_loss, acc_real, acc_fake):
    message = '''
               Epoch [{:5d}/{:5d}] | Batch [{:5d}/{:5d}] --> Rank {}
                  acc_real: {:6.4f} | d_loss: {:6.4f}
                  acc_fake: {:6.4f} | g_loss: {:6.4f}  
                '''.format(
                        epoch, config.num_epochs, batch_num, num_batches, rank,
                        acc_real, d_loss.item(), 
                        acc_fake, g_loss.item())
    
    with open(config.log_fname, 'a') as f:
        f.write(message)


# Write time to file
def log_time(config, epoch, start_time):
    message = f'\nTrain time for epoch {epoch}: {datetime.datetime.now()-start_time}\n' 

    with open(config.log_fname, 'a') as f:
        f.write(message)


# Clears file
def newfile(config):
    with open(config.log_fname, 'w') as _:
        pass

    with open(config.log_extra_fname, 'w') as _:
        pass


# Logs stuffs
def log_extra(config, rank, epoch, batch, info):
    acc_real, acc_fake, preds_real, preds_fake, av_real_pred, av_fake_pred = info
    preds_real = preds_real.squeeze()
    preds_fake = preds_fake.squeeze()
    nprint = 5
    message1 = f'''
Rank {rank} | epoch {epoch} | batch {batch} 
Acc. REAL: {round(acc_real, 2)} | Size: {preds_real.size()[0]} | Average Pred: {round(av_real_pred, 2)}: 
    {preds_real[:nprint].tolist()}
Acc. FAKE: {round(acc_fake, 2)} | Size: {preds_fake.size()[0]} | Average Pred: {round(av_fake_pred, 2)}: 
    {preds_fake[:nprint].tolist()}
    '''

    with open(config.log_extra_fname, 'a') as f:
        f.write(message1)