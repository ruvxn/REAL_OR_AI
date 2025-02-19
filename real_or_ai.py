import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

dataset_path = "dataset"
print("Dataset Folders:", os.listdir(dataset_path))

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        print(f"\nContents of {folder}: {os.listdir(folder_path)[:5]}")

def show_sample_images(category, num_images=5):
    category_path = os.path.join(dataset_path, category)
    images = random.sample(os.listdir(category_path), num_images)

    fig, axes = plt.subplots(1, num_images, figsize=(15,5))
    for i, img_name in enumerate(images):
        img = cv2.imread(os.path.join(category_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(category)
    plt.show()

categories = os.listdir(dataset_path)
for category in categories:
    if os.path.isdir(os.path.join(dataset_path, category)):
        show_sample_images(category)

train_csv_path = os.path.join(dataset_path, "train.csv")
train_df = pd.read_csv(train_csv_path)

train_df["file_name"] = train_df["file_name"].apply(lambda x: os.path.join(dataset_path, x))
print(train_df.head(20))

train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=524, stratify=train_df["label"])
print(f"Training set: {train_df.shape}")
print(f"Validation set: {val_df.shape}")

train_df["label"] = train_df["label"].astype(str)
val_df["label"] = val_df["label"].astype(str)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1.0/255.0,horizontal_flip=True,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.2)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,x_col="file_name", y_col="label",target_size=IMG_SIZE,batch_size=BATCH_SIZE,class_mode="binary")

val_generator = val_datagen.flow_from_dataframe(dataframe=val_df,x_col="file_name",y_col="label",target_size=IMG_SIZE,batch_size=BATCH_SIZE,class_mode="binary")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2)

history = model.fit(train_generator,validation_data=val_generator,epochs=15,callbacks=[early_stopping, reduce_lr])

plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

model.save("real_or_ai_classifier.h5")
print("Model saved as real_or_ai_classifier.h5")
