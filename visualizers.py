import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data


# Plot Gen/Disc losses through epochs
# Input is 2D numpy array where col0 is Gen and col1 is Disc
def plot_losses(config):
    losses_fname = config.losses_fname
    print(f'Plotting losses from: {losses_fname}')
    losses = np.load(losses_fname)

    num_epochs = losses.shape[0]
    x = np.arange(num_epochs)
    gen = losses[:, 0]
    disc = losses[:, 1]

    plt.figure()
    plt.plot(x, gen, '.-', label='Generator Loss')
    plt.plot(x, disc, '.-', label='Discriminator Loss')
    plt.title('Model Losses vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Binary Cross Entropy)')
    plt.legend()
    plt.show()


# View generator results though epochs
def view_samples(config):
    samples_fname = config.samples_fname
    print(f'Viewing samples from: {samples_fname}')

    samples = np.load(samples_fname)

    # Use this for epochs = 100
    every_n_epochs = 40
    cols = 8
    rows = int(len(samples)/every_n_epochs)
    print(len(samples), ' epochs in saved samples')
    _, axes = plt.subplots(figsize=(2*cols,2*rows), nrows=rows, ncols=cols, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for sample, ax_row in zip(samples[::every_n_epochs], axes):
        for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
            # print(f'Image min: {np.min(img)}, max: {np.max(img)}')
            # print(img.shape)
            if config.nchannels == 1: 
                img = img[0]
                ax.imshow(img, cmap='gray')

            else: 
                # Change (3, H, W) to (H, W, 3)
                img = np.transpose(img, (1, 2, 0))
                ax.imshow(img)

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    # Use this for viewing first 5 epochs
    # rows = 5
    # cols = 4
    # _, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

    # for sample, ax_row in zip(samples, axes):
    #     for img, ax in zip(sample, ax_row):
    #         img = img.detach().cpu()
    #         ax.imshow(img.permute(1,2,0))
    #         ax.xaxis.set_visible(False)
    #         ax.yaxis.set_visible(False)

    
    plt.show()


# View samples from dataset
def view_dataset(config, data_path):
    print('Viewing samples from dataset')
    dataset = load_data(config, data_path, 0, 0)
    for batch, _ in dataset: break

    rows = 4
    cols = 4
    _, axes = plt.subplots(figsize=(2*cols,2*rows), nrows=rows, ncols=cols, sharex=True, sharey=True)
    count = 0
    for axes_row in axes:
        for ax in axes_row:
            img = batch[count]
            if config.nchannels == 1: 
                img = img[0]
                ax.imshow(img, cmap='gray')

            else: 
                img = np.transpose(img, (1, 2, 0))
                ax.imshow(img)

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            count += 1
    plt.show()

# Plot Discriminator accuracy
def plot_accuracy(config):
    acc_fname = config.acc_fname
    print(f'Plotting accuracies from: {acc_fname}')
    accuracies = np.load(acc_fname)

    real_acc = accuracies[:, 0]
    fake_acc = accuracies[:, 1]
    av_acc = np.mean(accuracies, axis=1)
    num_epochs = accuracies.shape[0]
    epochs = np.arange(num_epochs)

    plt.figure()
    plt.plot(epochs, real_acc, '.-', label='Acc. on Real')
    plt.plot(epochs, fake_acc, '.-', label='Acc. on Fake')
    plt.plot(epochs, av_acc, '.-', label='Average')
    plt.legend()
    ymax = max(1, np.max(accuracies)) + 0.05
    plt.ylim(-0.05, ymax)
    plt.title('Discriminator Accuracy')
    plt.show()


# Plot losses and accuracy together
def plot_together(config):
    losses_fname = config.losses_fname
    acc_fname = config.acc_fname
    av_preds_fname = config.av_preds_fname
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    
    print(f'Plotting losses from: {losses_fname}')
    losses = np.load(losses_fname)
    num_epochs = losses.shape[0]
    x = np.arange(num_epochs)
    gen = losses[:, 0]
    disc = losses[:, 1]

    ax1.plot(x, gen, '.-', label='Generator Loss')
    ax1.plot(x, disc, '.-', label='Discriminator Loss')
    ax1.set_title('Losses')
    ax1.legend()
    ax1.set_ylabel('BCE Loss')
    ax1.grid(True)

    print(f'Plotting accuracies from: {acc_fname}')
    accuracies = np.load(acc_fname)
    real_acc = accuracies[:, 0]
    fake_acc = accuracies[:, 1]
    av_acc = np.mean(accuracies, axis=1)

    ax2.plot(x, real_acc, '.-', label='Acc. on Real')
    ax2.plot(x, fake_acc, '.-', label='Acc. on Fake')
    ax2.plot(x, av_acc, '.-', label='Average')
    ax2.legend()
    ymax = max(1, np.max(accuracies)) + 0.05
    ax2.set_ylim(-0.05, ymax)
    ax2.set_title('Discriminator Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    print(f'Plotting average predictions from: {acc_fname}')
    av_preds = np.load(av_preds_fname)
    av_pred_real = av_preds[:, 0]
    av_pred_fake = av_preds[:, 1]

    ax3.plot(x, av_pred_real, '.-', label='Av. Pred on Real')
    ax3.plot(x, av_pred_fake, '.-', label='Av. Pred on Fake')
    ax3.legend()
    ymax = max(1, np.max(accuracies)) + 0.05
    ax3.set_ylim(-0.05, ymax)
    ax3.set_title('Discriminator Average Prediciton')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Prediciton')
    ax3.grid(True)

    plt.show()