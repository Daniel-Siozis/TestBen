import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
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
def load_data():
    # Hier lädst du deine Bilder und ihre beschädigten Versionen
    train_images_paths = ['Dataset/9908/Probe/korrekt/9908_korrekt_2_f000000.jpg',
                    'Dataset/9908/Probe/korrekt/9908_korrekt_2_f000001.jpg',
                     'Dataset/9908/Probe/korrekt/9908_korrekt_2_f000002.jpg',
                      'Dataset/9908/Probe/korrekt/9908_korrekt_2_f000003.jpg',
                       'Dataset/9908/Probe/korrekt/9908_korrekt_2_f000004.jpg',
                        'Dataset/9908/Probe/korrekt/9908_korrekt_2_f000005.jpg',
                         'Dataset/9908/Probe/korrekt/9908_korrekt_2_f000006.jpg',
                          'Dataset/9908/Probe/korrekt/9908_korrekt_2_f000007.jpg',
                           'Dataset/9908/Probe/korrekt/9908_korrekt_2_f000008.jpg',
                            'Dataset/9908/Probe/korrekt/9908_korrekt_2_f000009.jpg',
                             'Dataset/9908/Probe/korrekt/9908_korrekt_2_f000010.jpg']  # Liste mit den 10 Bildern
    train_images_noisy_paths = ['Dataset/9908/Probe/broken/9908_broken_2_f000000.jpg',
                    'Dataset/9908/Probe/broken/9908_broken_2_f000001.jpg',
                     'Dataset/9908/Probe/broken/9908_broken_2_f000002.jpg',
                      'Dataset/9908/Probe/broken/9908_broken_2_f000003.jpg',
                       'Dataset/9908/Probe/broken/9908_broken_2_f000004.jpg',
                        'Dataset/9908/Probe/broken/9908_broken_2_f000005.jpg',
                         'Dataset/9908/Probe/broken/9908_broken_2_f000006.jpg',
                          'Dataset/9908/Probe/broken/9908_broken_2_f000007.jpg',
                           'Dataset/9908/Probe/broken/9908_broken_2_f000008.jpg',
                            'Dataset/9908/Probe/broken/9908_broken_2_f000009.jpg',
                             'Dataset/9908/Probe/broken/9908_broken_2_f000010.jpg']   # Liste mit den beschädigten Versionen der 10 Bilder
    train_images = []
    for path in train_images_paths:
        image = Image.open(path)
        image_array = np.array(image)
        train_images.append(image_array)

    train_images_noisy = []
    for path in train_images_noisy_paths:
        noisy_image = Image.open(path)
        noisy_image_array = np.array(noisy_image)
        train_images_noisy.append(noisy_image_array)

    # Konvertiere die Listen in NumPy-Arrays und füge die Batch-Dimension hinzu
    train_array_correct_dim = np.array(train_images)
    train_array_broken_dim = np.array(train_images_noisy)

    print(train_array_correct_dim.shape)
    print(train_array_broken_dim.shape)
    return train_array_correct_dim, train_array_broken_dim

# Definiere den Modelltrainingsprozess
def train_model(model, train_images, train_images_noisy):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    model.fit(train_images_noisy, train_images, epochs=10, batch_size=1)

# Hauptfunktion zum Trainieren und Speichern des Modells
def main():
    # Lade Daten
    train_images, train_images_noisy = load_data()

    # Erstelle das Modell
    model = create_model()

    # Trainiere das Modell
    train_model(model, train_images, train_images_noisy)

    # Speichere das trainierte Modell
    model.save('trained_model.h5')

if __name__ == "__main__":
    main()
