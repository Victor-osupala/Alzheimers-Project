import os
import numpy as np
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 10

# Dataset paths
train_dir = "Combined Dataset/train"
val_dir = "Combined Dataset/test"

# Class to label mapping
label_map = {
    "No Impairment": 0,
    "Very Mild Impairment": 1,
    "Mild Impairment": 2,
    "Moderate Impairment": 3
}

# === LOAD IMAGES FROM DIRECTORY ===
def load_images_from_directory(directory):
    images = []
    labels = []
    for label_name, label_idx in label_map.items():
        folder_path = os.path.join(directory, label_name)
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: {folder_path} does not exist.")
            continue

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                img = np.expand_dims(img, axis=-1)
                images.append(img)
                labels.append(label_idx)

    return np.array(images), to_categorical(np.array(labels), num_classes=NUM_CLASSES)

# Load training and validation data
X_train, y_train = load_images_from_directory(train_dir)
X_val, y_val = load_images_from_directory(val_dir)

print(f"✅ Loaded {len(X_train)} training images and {len(X_val)} validation images.")

# === DEFINE CNN MODEL ===
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train model
model = build_model()
model.summary()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save model
model.save("alzheimer_image_classifier.h5")
print("✅ Model trained and saved as 'alzheimer_image_classifier.h5'")
