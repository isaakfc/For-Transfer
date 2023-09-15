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
    plt.savefig('PLOTS/gan_loss_epoch_%d.png' % epoch)
    plt.close()

