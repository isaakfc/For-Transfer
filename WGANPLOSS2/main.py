import load
import tensorflow as tf
import autoencoder
import discriminator
import plot
import train
import numpy as np
#tf.config.set_visible_devices([], 'GPU')

# =========================
# DATA PREPARATION
# =========================

# Paths for data
SPECTROGRAMS_PASSAGES_PATH = "/Users/isaac/Desktop/TempData/SpectrogramsPassages"
SPECTROGRAMS_RECREATIONS_PATH = "/Users/isaac/Desktop/TempData/SpectrogramsRecreations"

# Load spectrograms
x_train = load.load_data_set(SPECTROGRAMS_PASSAGES_PATH)
x_noisy = load.load_data_set(SPECTROGRAMS_RECREATIONS_PATH)

x_train = x_train.astype('float32')
x_noisy = x_noisy.astype('float32')

print(x_train.shape)

# =========================
# HYPER PARAMETERS
# =========================

LEARNING_RATE = 2e-4    # Range from about  0.0001 to 0.1
MIN_LR = 0.000001 # Minimum value of learning rate
DECAY_FACTOR=1.00004 # learning rate decay factor
EPOCHS = 50             # Range from about 5 to 100
L2_REG = 0.005           # Range from about 0 to 0.01
BATCH_SIZE = 10      # Range from 16 to 512 (for audio from about 8 to 128)
#FILTERS = [32, 32, 16]  # better if it is [64, 64, 32] or [128, 128, 64] for audio
FILTERS = [64, 64, 32]
ALPHA = 0.1
BETA_1 = 0.5
MOMENTUM = 0.8
GAMMA = 1
PRINT_FREQ = 100
LAMBDA = 10 # For gradient penalty
N_CRITIC = 3 # Train critic(discriminator) n times then train generator 1 time.

# =========================
# TRAINING SET-UP
# =========================

# Initialise models
generatorModel = autoencoder.Autoencoder(FILTERS, L2_REG)
discriminatorModel = discriminator.Discriminator(ALPHA)

# Initialise optimizers
generator_optimizer = tf.keras.optimizers.legacy.Adam(LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.legacy.Adam(LEARNING_RATE, beta_1=BETA_1)

batch_count = int(x_train.shape[0] / BATCH_SIZE)
print('Epochs:', EPOCHS)
print('Batch size:', BATCH_SIZE)
print('Batches per epoch:', batch_count)

# To be used for plot
dLosses = []
gTotalLosses = []
gLosses = []
gRecLosses = []

current_learning_rate = LEARNING_RATE
trace = True
n_critic_count = 0

# =========================
# MAIN TRAINING LOOP
# =========================

for e in range(1, EPOCHS+1):
    print('\n', '-' * 15, 'Epoch %d' % e, '-' * 15)

    current_learning_rate = train.learning_rate_decay(current_learning_rate, DECAY_FACTOR, MIN_LR)
    print('current_learning_rate %.10f' % (current_learning_rate,))
    train.set_learning_rate(current_learning_rate, discriminator_optimizer, generator_optimizer)

    # Shuffle training data
    permutation = np.random.permutation(x_train.shape[0])
    x_train_shuffled = x_train[permutation]
    x_train_noisy_shuffled = x_noisy[permutation]

    for batch_idx in range(batch_count):
        # Get start and end index
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        # Extract from shuffled data
        x_real_batch = x_train_shuffled[start_idx:end_idx]
        x_noisy_batch = x_train_noisy_shuffled[start_idx:end_idx]

        # Train discriminator model
        d_loss = train.WGAN_GP_train_d_step(x_real_batch,
                                             x_noisy_batch,
                                             discriminatorModel,
                                             generatorModel,
                                             discriminator_optimizer,
                                             LAMBDA,
                                             BATCH_SIZE)
        n_critic_count += 1

        if n_critic_count >= N_CRITIC:
            # Train generator
            total_g_loss, r_loss, g_loss, generated_images = train.WGAN_GP_train_g_step(x_real_batch,
                                                                                        x_noisy_batch,
                                                                                        discriminatorModel,
                                                                                        generatorModel,
                                                                                        generator_optimizer,
                                                                                        GAMMA)
            n_critic_count = 0


        # Print progress
        #if (batch_idx + 1) % PRINT_FREQ == 0:
            #print(
                #f"\rEpoch {e}, Batch {batch_idx + 1}/{batch_count}: Discriminator Loss: {d_loss:.4f}, /"
                #f"Total generator Loss: {total_g_loss:.4f}, Generator Loss: {g_loss:.4f}, /"
                #f"Reconstruction Loss: {r_loss:.4f}")

        #else:
            #print(f"\rBatch {batch_idx + 1}/{batch_count}", end='')

        print(f"\rEpoch {e}, Batch {batch_idx + 1}/{batch_count}")


    dLosses.append(d_loss)
    gLosses.append(g_loss)
    gRecLosses.append(r_loss)
    gTotalLosses.append(total_g_loss)

#plot.plot_loss(e, dLosses, gTotalLosses, gRecLosses, gLosses)


# Start new line
print()
