import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models, datasets, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from vae import *

parser = argparse.ArgumentParser()
parser.add_argument("--load_vae", action='store_true', default=False)
parser.add_argument("--load_classifier", action='store_true', default=False)
parser.add_argument("--train_vae", action='store_true', default=False)
parser.add_argument("--train_classifier", action='store_true', default=False)

args = parser.parse_args()

BATCH_SIZE = 32
EPOCHS = 20

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

encoder = create_encoder()
decoder = create_decoder()

if args.load_vae:
    encoder.load_weights("results/encoder.weights.h5")
    decoder.load_weights("results/decoder.weights.h5")
    
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

if args.train_vae:
    noise_factor = 0.1
    x_train_noisy = x_train + np.random.normal(loc=0.0, scale=noise_factor, size=x_train.shape)
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)

    history = vae.fit(
        x_train_noisy,
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

    encoder.save_weights("results/encoder.weights.h5")
    decoder.save_weights("results/decoder.weights.h5")
    

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

# Загрузка тестовых данных
(_, _), (X_test, y_test) = datasets.cifar10.load_data()
X_test = X_test.astype('float32') / 255
y_test = tf.keras.utils.to_categorical(y_test, 10)

if args.load_classifier:
    encoder.load_weights("results/classifier.weights.h5")
else:
    # Генерация синтетических данных
    # Вычисление средних векторов классов
    class_means = []
    for class_idx in range(10):
        class_images = x_train[y_train.flatten() == class_idx]
        z_means, _, _ = encoder.predict(class_images, verbose=0)
        class_means.append(np.mean(z_means, axis=0))

    # Генерация 5000 примеров на класс
    GEN_BATCH_SIZE = 1000
    GEN_AMOUNT     = 5000

    X_gen = []
    y_gen = []

    for class_idx in range(10):
        print(f"Генерация класса {class_idx}...")
        z_mean = class_means[class_idx]

        mean_image = decoder.predict(z_mean.reshape(1, -1), verbose=0)[0]
        # Конвертируем и сохраняем
        mean_image = (mean_image * 255).astype(np.uint8)
        Image.fromarray(mean_image).save(f"results/class_{class_idx}.png")

        # Генерация батчами для экономии памяти
        for _ in range(GEN_AMOUNT // GEN_BATCH_SIZE):
            noise = np.random.normal(scale=0.5, size=(GEN_BATCH_SIZE, LATENT_DIM))
            generated = decoder.predict(z_mean + noise, verbose=0)
            X_gen.append(generated)
            y_gen.extend([class_idx]*GEN_BATCH_SIZE)

    X_train_gen = np.vstack(X_gen)
    y_train_gen = tf.keras.utils.to_categorical(y_gen, 10)

    # Визуализация примеров
    plt.figure(figsize=(10,10))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(X_train_gen[(i // 10)*GEN_AMOUNT + i % 10])
        plt.axis('off')
    plt.suptitle('Примеры сгенерированных изображений')
    plt.savefig('generated_samples.png', bbox_inches='tight')
    plt.close()

    # =============================================
    # Часть 3: Обучение классификатора
    # =============================================

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

    encoder.save_weights("results/classifier.weights.h5")

# Оценка
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nТочность на реальных данных: {test_acc * 100:.2f}%')
