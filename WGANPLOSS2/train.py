import tensorflow as tf
from keras import backend as K
import openl3
import numpy as np

input_repr, content_type, embedding_size = 'linear', 'music', 6144
model = openl3.models.load_audio_embedding_model(input_repr, content_type, embedding_size, frontend='librosa')

def normalize_and_convert_to_rgb_tf(spectrogram_tensor):
    # Normalize the spectrogram tensor
    max_val = tf.reduce_max(spectrogram_tensor, axis=(1, 2), keepdims=True)
    normalized_spectrogram = spectrogram_tensor / (max_val + 1e-5)

    # Convert normalized data to range [0, 255]
    scaled_spectrogram = tf.cast(normalized_spectrogram * 255, tf.uint8)

    # Convert to RGB by repeating across the last dimension
    return tf.repeat(scaled_spectrogram, 3, axis=-1)


@tf.function
def WGAN_GP_train_d_step(real_x,
                         noisy_x,
                         discriminator,
                         generator,
                         discriminator_optimizer,
                         LAMBDA,
                         batch_size):

    epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
    ###################################
    # Train D
    ###################################
    with tf.GradientTape(persistent=True) as d_tape:
        with tf.GradientTape() as gp_tape:
            fake_image = generator(noisy_x, training=True)
            fake_image_mixed = epsilon * tf.dtypes.cast(real_x, tf.float32) + ((1 - epsilon) * fake_image)
            fake_mixed_pred = discriminator(fake_image_mixed, training=True)

        # Compute gradient penalty
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))

        fake_pred = discriminator(fake_image, training=True)
        real_pred = discriminator(real_x, training=True)

        D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
    # Calculate the gradients for discriminator
    D_gradients = d_tape.gradient(D_loss,
                                  discriminator.trainable_variables)
    # Apply the gradients to the optimizer
    discriminator_optimizer.apply_gradients(zip(D_gradients,
                                    discriminator.trainable_variables))

    return D_loss

@tf.function
def WGAN_GP_train_g_step(real_x, noisy_x, discriminator, generator, generator_optimizer, gamma=0.5, beta=0.1):
    ###################################
    # Train G
    ###################################
    with tf.GradientTape() as g_tape:
        fake_image = generator(noisy_x, training=True)
        fake_pred = discriminator(fake_image, training=True)

        # Get embeddings
        real_emb, fake_emb = get_audio_embeddings(real_x, fake_image)
        # Compute perceptual loss
        percep_loss = tf.reduce_mean(tf.square(real_emb - fake_emb))  # MSE

        # Compute reconstruction loss
        recon_loss = tf.reduce_mean(tf.square(real_x - fake_image))  # MSE

        # Compute generative loss
        g_loss = -tf.reduce_mean(fake_pred)

        # Combined loss
        total_g_loss = -tf.reduce_mean(fake_pred) + gamma * recon_loss + beta * percep_loss

    # Calculate the gradients for generator
    g_gradients = g_tape.gradient(total_g_loss,
                                  generator.trainable_variables)
    # Apply the gradients to the optimizer
    generator_optimizer.apply_gradients(zip(g_gradients,
                                            generator.trainable_variables))

    return total_g_loss, recon_loss, g_loss, fake_image


def learning_rate_decay(current_lr, decay_factor, MIN_LR):
    '''
        Calculate new learning rate using decay factor
    '''
    new_lr = max(current_lr / decay_factor, MIN_LR)
    return new_lr

def set_learning_rate(new_lr, D_optimizer, G_optimizer):
    '''
        Set new learning rate to optimizers
    '''
    K.set_value(D_optimizer.lr, new_lr)
    K.set_value(G_optimizer.lr, new_lr)


def get_audio_embeddings(real_x, fake_image):
    real_emb = model(real_x, training=False)
    fake_emb = model(fake_image, training=False)

    return real_emb, fake_emb

