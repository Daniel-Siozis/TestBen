import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import os
from PIL import Image

# Definiere dein vortrainiertes Modell, z.B. ein einfaches CNN
def create_model():
    model = Sequential()
    model.add(Input(shape=(1080, 1920, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(3, (3, 3), activation='relu', padding='same'))
    return model

# Lade deine 10 Bilder und ihre beschädigten Versionen
def load_data(train_path, noisy_path):
    # Hier lädst du deine Bilder und ihre beschädigten Versionen
    valid_images_paths = []  # Liste mit den 10 Bildern
    broken_images_paths = []   # Liste mit den beschädigten Versionen der 10 Bilder

    for filename in os.listdir(train_path):
        img_path = os.path.join(train_path, filename)
        valid_images_paths.append(img_path)

    for filename in os.listdir(noisy_path):
        img_path = os.path.join(noisy_path, filename)
        broken_images_paths.append(img_path)

    valid_images = []
    for path in valid_images_paths:
        image = Image.open(path)
        valid_images.append(np.array(image))

    broken_images = []
    for path in broken_images_paths:
        noisy_image = Image.open(path)
        broken_images.append(np.array(noisy_image))

    # Konvertiere die Listen in NumPy-Arrays und füge die Batch-Dimension hinzu
    valid_train_images = np.array(valid_images)
    broken_train_images = np.array(broken_images)

    print(valid_train_images.shape)
    print(broken_train_images.shape)
    return valid_train_images, broken_train_images

# Definiere den Modelltrainingsprozess
def train_model(model, train_images, train_images_noisy):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    model.fit(train_images_noisy, train_images, epochs=10, batch_size=1)

# Hauptfunktion zum Trainieren und Speichern des Modells
def main():
    # Lade Daten
    train_path = 'Dataset\lid'
    noisy_path = 'Dataset\oken'
    train_images, train_images_noisy = load_data(train_path, noisy_path)

    # Erstelle das Modell
    model = create_model()

    # Trainiere das Modell
    train_model(model, train_images, train_images_noisy)

    # Speichere das trainierte Modell
    model.save('trained_model.h5')

if __name__ == "__main__":
    main()
