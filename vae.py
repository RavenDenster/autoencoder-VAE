import keras
import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE=32
KL_LOSS_FACTOR = 0.5
LATENT_DIM = 128

# Слой сэмплирования
@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))  # N(0, 1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@keras.saving.register_keras_serializable()
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        noisy_inputs, original_images = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(noisy_inputs)
            reconstruction = self.decoder(z)
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(original_images, reconstruction),
                    axis=(1, 2)
                )
            )
            
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )

            total_loss = reconstruction_loss + KL_LOSS_FACTOR * kl_loss 
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "rec_loss": reconstruction_loss, "kl_loss": kl_loss}

def create_encoder():
    encoder_inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def create_decoder():
    latent_inputs = keras.Input(shape=(LATENT_DIM,))
    x = layers.Dense(8*8*64)(latent_inputs)
    x = layers.Reshape((8, 8, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    decoder_outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")
