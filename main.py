import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA

IMG_SIZE = 32
LATENT_DIM = 128
BATCH_SIZE = 256
EPOCHS = 50

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

encoder_inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Flatten()(x)
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

latent_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(8*8*64, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 64))(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
decoder_outputs = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            ) * IMG_SIZE * IMG_SIZE
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "rec_loss": reconstruction_loss, "kl_loss": kl_loss}

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
history = vae.fit(
    x_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE
)

os.makedirs("results", exist_ok=True)
plt.plot(history.history['loss'], label='Total loss')
plt.plot(history.history['rec_loss'], label='Reconstruction loss')
plt.plot(history.history['kl_loss'], label='KL loss')
plt.legend()
plt.title('Training dynamics')
plt.savefig("results/training_history.png", bbox_inches='tight', dpi=150)
plt.close()

os.makedirs("results", exist_ok=True)
z_means, _, _ = encoder.predict(x_train, verbose=0)
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(z_means)

plt.figure(figsize=(10, 8))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y_train.squeeze(), cmap='tab10', alpha=0.6)
plt.colorbar(label='Class')
plt.title('Latent space (2D PCA)')
plt.savefig("results/latent_space_pca.png", bbox_inches='tight', dpi=150)
plt.close()

class_means = []
for class_idx in range(10):
    class_images = x_train[y_train.flatten() == class_idx]
    _, z_means, _ = encoder.predict(class_images, verbose=0)
    class_means.append(np.mean(z_means, axis=0))

def save_comparison(original_images, generated_images, class_idx, save_dir="results"):
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Оригиналы")
    combined_original = original_images.transpose(1, 0, 2, 3).reshape(32, -1, 3)
    plt.imshow(combined_original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Сгенерированные")
    combined_generated = generated_images.transpose(1, 0, 2, 3).reshape(32, -1, 3)
    plt.imshow(combined_generated)
    plt.axis('off')

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        f"{save_dir}/class_{class_idx}_comparison.png",
        bbox_inches='tight',
        dpi=150
    )
    plt.close()

for class_idx in range(10):
    class_indices = np.where(y_train.flatten() == class_idx)[0]
    original_samples = x_train[np.random.choice(class_indices, 10)]

    z_mean = class_means[class_idx]
    noise = np.random.normal(scale=0.1, size=(10, LATENT_DIM))
    generated_samples = decoder.predict(z_mean + noise, verbose=0)

    save_comparison(original_samples, generated_samples, class_idx)
