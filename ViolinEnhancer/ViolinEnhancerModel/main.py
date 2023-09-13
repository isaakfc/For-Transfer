import load
import tensorflow as tf
import autoencoder
import discriminator
import plot
import train
import numpy as np
import audiogeneration

tf.config.set_visible_devices([], 'GPU')

# =========================
# DATA PREPARATION
# =========================

# Paths for data
SPECTROGRAMS_PASSAGES_PATH = "/Users/isaac/Documents/DATASETS/SAMPLE_ENHANCEMENT/SPECTROGRAMS_PASSAGES"
SPECTROGRAMS_RECREATIONS_PATH = "/Users/isaac/Documents/DATASETS/SAMPLE_ENHANCEMENT/SPECTROGRAMS_RECREATIONS"

# Load spectrograms
x_clean = load.load_data_set(SPECTROGRAMS_PASSAGES_PATH)[:200]
x_noisy = load.load_data_set(SPECTROGRAMS_RECREATIONS_PATH)[:200]

# Convert to 32 bit floating point
x_clean = x_clean.astype('float32')
x_noisy = x_noisy.astype('float32')

# Normalise
overall_min = min(np.min(x_clean), np.min(x_noisy))
overall_max = max(np.max(x_clean), np.max(x_noisy))

x_clean = (x_clean - overall_min) / (overall_max - overall_min)
x_noisy = (x_noisy - overall_min) / (overall_max - overall_min)

# Get the indicies from the training array, shuffle and limit to 50
indices = np.arange(x_clean.shape[0])
np.random.shuffle(indices)
test_indices = indices[:50]

# Take the corrosponding random indicies from the training arrays to make test set
x_clean_test = x_clean[test_indices]
x_noisy_test = x_noisy[test_indices]

# Remove the test data from training set
x_clean_train = np.delete(x_clean, test_indices, axis=0)
x_noisy_train = np.delete(x_noisy, test_indices, axis=0)

# =========================
# HYPER PARAMETERS
# =========================

GENERATOR_LEARNING_RATE = 2e-4    # Range from about  0.0001 to 0.1
DISCRIMINATOR_LEARNING_RATE = 2e-4    # Range from about  0.0001 to 0.1
MIN_LR = 0.000001 # Minimum value of learning rate
DECAY_FACTOR = 1.00004 # learning rate decay factor
EPOCHS = 1             # Range from about 5 to 100
L2_REG = 0.005           # Range from about 0 to 0.01
BATCH_SIZE = 15     # Range from 16 to 512 (for audio from about 8 to 128)
#FILTERS = [32, 32, 16]  # better if it is [64, 64, 32] or [128, 128, 64] for audio
FILTERS = [64, 64, 32]
ALPHA = 0.1
BETA_1 = 0.5
MOMENTUM = 0.8
GAMMA = 1
PRINT_FREQ = 10
LAMBDA = 10  # For gradient penalty
N_CRITIC = 3  # Train critic(discriminator) n times then train generator 1 time.

# =========================
# GLOBAL PARAMETERS
# =========================

SR = 22000
HOP_LENGTH = 256
FRAME_SIZE = 512

# =========================
# TRAINING SET-UP
# =========================

# Initialise models
generatorModel = autoencoder.Autoencoder(FILTERS, L2_REG)
discriminatorModel = discriminator.Discriminator(ALPHA)

# Initialise optimizers
generator_optimizer = tf.keras.optimizers.legacy.Adam(GENERATOR_LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.legacy.Adam(DISCRIMINATOR_LEARNING_RATE, beta_1=BETA_1)

batch_count = int(x_clean.shape[0] / BATCH_SIZE)
print('Epochs:', EPOCHS)
print('Batch size:', BATCH_SIZE)
print('Batches per epoch:', batch_count)

# To be used for plot
dLosses = []
gTotalLosses = []
gLosses = []
gRecLosses = []

trace = True
n_critic_count = 0

# =========================
# MAIN TRAINING LOOP
# =========================

for e in range(1, EPOCHS+1):
    print('\n', '-' * 15, 'Epoch %d' % e, '-' * 15)

    # Adjust learning rates
    DISCRIMINATOR_LEARNING_RATE = train.learning_rate_decay(DISCRIMINATOR_LEARNING_RATE, DECAY_FACTOR, MIN_LR)
    GENERATOR_LEARNING_RATE = train.learning_rate_decay(GENERATOR_LEARNING_RATE, DECAY_FACTOR, MIN_LR)
    print('current_learning_rate_discriminator %.10f' % (DISCRIMINATOR_LEARNING_RATE,))
    print('current_learning_rate_generator %.10f' % (GENERATOR_LEARNING_RATE,))
    train.set_learning_rate(DISCRIMINATOR_LEARNING_RATE, discriminator_optimizer)
    train.set_learning_rate(GENERATOR_LEARNING_RATE, generator_optimizer)

    # Shuffle training data
    permutation = np.random.permutation(x_clean.shape[0])
    x_train_shuffled = x_clean[permutation]
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
            total_g_loss, r_loss, g_loss, generated_spectrograms = train.WGAN_GP_train_g_step(x_real_batch,
                                                                                              x_noisy_batch,
                                                                                              discriminatorModel,
                                                                                              generatorModel,
                                                                                              generator_optimizer,
                                                                                              GAMMA)
            x_real_batch_for_plot = x_real_batch
            x_noisy_batch_for_plot = x_noisy_batch
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

    if e == 1 or e % 5 == 0:
        plot.saveGeneratedSpectrograms(e, generated_spectrograms, x_real_batch_for_plot, x_noisy_batch_for_plot)
        audiogeneration.convert_spectrograms_to_audio(e,
                                                      generated_spectrograms,
                                                      overall_max,
                                                      overall_min,
                                                      HOP_LENGTH,
                                                      FRAME_SIZE,
                                                      SR,
                                                      num_files=5,
                                                      file_prefix="network_generated_audio")
        audiogeneration.convert_spectrograms_to_audio(e,
                                                      x_real_batch,
                                                      overall_max,
                                                      overall_min,
                                                      HOP_LENGTH,
                                                      FRAME_SIZE,
                                                      SR,
                                                      num_files=1,
                                                      file_prefix="passage_test_audio",
                                                      output_dir="INPUT_AUDIO_TESTS")
        audiogeneration.convert_spectrograms_to_audio(e,
                                                       x_noisy_batch,
                                                       overall_max,
                                                       overall_min,
                                                       HOP_LENGTH,
                                                       FRAME_SIZE,
                                                       SR,
                                                       num_files=1,
                                                       file_prefix="samples_test_audio",
                                                       output_dir="INPUT_AUDIO_TESTS")

    dLosses.append(d_loss)
    gLosses.append(g_loss)
    gRecLosses.append(r_loss)
    gTotalLosses.append(total_g_loss)

plot.plot_loss(e, dLosses, gTotalLosses, gRecLosses, gLosses)


# Start new line
print()
