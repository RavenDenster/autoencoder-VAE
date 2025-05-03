import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models, datasets, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =============================================
# Часть 1: Обучение вариационного автоэнкодера
# =============================================

# Параметры
IMG_SIZE = 32
LATENT_DIM = 256
BATCH_SIZE = 128
EPOCHS = 50

# Загрузка данных
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Слой сэмплирования
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Энкодер
encoder_inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Flatten()(x)
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Декодер
latent_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(8*8*256, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 256))(x)
x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
decoder_outputs = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# VAE модель
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
            total_loss = reconstruction_loss + 0.5 * kl_loss 
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "rec_loss": reconstruction_loss, "kl_loss": kl_loss}

# Обучение VAE
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

# =============================================
# Часть 2: Генерация синтетических данных
# =============================================

# Вычисление средних векторов классов
class_means = []
for class_idx in range(10):
    class_images = x_train[y_train.flatten() == class_idx]
    _, z_means, _ = encoder.predict(class_images, verbose=0)
    class_means.append(np.mean(z_means, axis=0))


# Генерация 5000 примеров на класс
num_samples_per_class = 5000
X_gen = []
y_gen = []

for class_idx in range(10):
    print(f"Генерация класса {class_idx}...")
    z_mean = class_means[class_idx]
    # Генерация батчами по 1000 для экономии памяти
    for _ in range(5):
        noise = np.random.normal(scale=0.05, size=(1000, LATENT_DIM))
        generated = decoder.predict(z_mean + noise, verbose=0)
        X_gen.append(generated)
        y_gen.extend([class_idx]*1000)

X_train_gen = np.vstack(X_gen)
y_train_gen = tf.keras.utils.to_categorical(y_gen, 10)

# =============================================
# Часть 3: Обучение классификатора
# =============================================

# Загрузка тестовых данных
(_, _), (X_test, y_test) = datasets.cifar10.load_data()
X_test = X_test.astype('float32') / 255
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Создание модели
model = models.Sequential([
    layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.4),

    layers.Conv2D(256, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(256, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.5),

    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Компиляция
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Аугментация
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(X_train_gen)

# Ранняя остановка
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

# Обучение
history = model.fit(
    datagen.flow(X_train_gen, y_train_gen, batch_size=64),
    epochs=25,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Оценка
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nТочность на реальных данных: {test_acc * 100:.2f}%')

# =============================================
# Ожидаемые результаты:
# 1. При обучении на оригинальных данных (как в ЛР4) точность была около ~85%
# 2. При обучении на синтетических данных точность будет ниже (~60-75% в зависимости от качества VAE)
# 3. Это показывает, что сгенерированные данные уступают реальным по информативности
# 4. Визуальный анализ сгенерированных изображений может показать размытость и недостаток деталей
# =============================================

# Визуализация примеров
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train_gen[i])
    plt.axis('off')
plt.suptitle('Примеры сгенерированных изображений')
plt.savefig('generated_samples.png', bbox_inches='tight')
plt.close()