import matplotlib.pyplot as plt
import numpy as np

def plot_loss(epoch, dLosses, gLosses, rLosses, standalone_gLosses):
    plt.figure(figsize=(10, 8))

    # Plotting Discriminative loss (bolder)
    plt.plot(dLosses, label='Discriminative Loss', linewidth=2, color='magenta')

    # Plotting Total Generative loss (bolder)
    plt.plot(gLosses, label='Total Generative Loss', linewidth=2, color='green')

    # Plotting Reconstruction loss (thinner, same color, and slightly transparent)
    plt.plot(rLosses, label='Reconstruction Loss', color='orange', alpha=0.6)

    # Plotting standalone Generative loss (thinner, same color, and slightly transparent)
    plt.plot(standalone_gLosses, label='Generator Loss', color='blue', alpha=0.6)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('LOSSPLOTS/gan_loss_epoch_%d.png' % epoch)
    plt.close()


import matplotlib.pyplot as plt


def saveGeneratedSpectrograms(epoch, clean_spectrograms,generated_spectrograms , noisy_spectrograms, dim=(3, 10), figsize=(10, 3)):

    num_to_plot = min(10, generated_spectrograms.shape[0])
    plt.figure(figsize=figsize)

    # Plotting clean spectrograms
    for i in range(num_to_plot):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(clean_spectrograms[i, :, :, 0], interpolation='nearest',
                   cmap='inferno', vmin=0, vmax=1, aspect='auto', origin='lower')
        plt.axis('off')

    # Plotting noisy spectrograms
    for i in range(num_to_plot):
        plt.subplot(dim[0], dim[1], i + 1 + dim[1])  # Shift by 10 to move to next row
        plt.imshow(noisy_spectrograms[i, :, :, 0], interpolation='nearest',
                   cmap='inferno', vmin=0, vmax=1, aspect='auto', origin='lower')
        plt.axis('off')

    # Plotting generated spectrograms
    for i in range(num_to_plot):
        plt.subplot(dim[0], dim[1], i + 1 + 2*dim[1])  # Shift by 20 to move to third row
        plt.imshow(generated_spectrograms[i, :, :, 0], interpolation='nearest',
                   cmap='inferno', vmin=0, vmax=1, aspect='auto', origin='lower')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'SPECTROGRAMPLOTS/generated_image_epoch_{epoch}.png')
    plt.close()



