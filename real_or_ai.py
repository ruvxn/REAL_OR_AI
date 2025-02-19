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

# Define dataset path
dataset_path = "dataset"
print("Dataset Folders:", os.listdir(dataset_path))

# Print sample files from each category
def show_sample_images(category, num_images=5):
    category_path = os.path.join(dataset_path, category)
    images = random.sample(os.listdir(category_path), num_images)
    
    fig, axes = plt.subplots(1, num_images, figsize=(15,5))
    for i, img_name in enumerate(images):
        img = cv2.imread(os.path.join(category_path, img_name))  # Read image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format (The primary difference between BGR and RGB is the order in which the color channels (red, green, and blue) are arranged within a pixel)
        axes[i].imshow(img)
        axes[i].axis("off")  
        axes[i].set_title(category) 
    plt.show()

# Display sample images from each category
categories = os.listdir(dataset_path)
for category in categories:
    if os.path.isdir(os.path.join(dataset_path, category)):
        show_sample_images(category)

# Load dataset CSV file into a DataFrame
train_csv_path = os.path.join(dataset_path, "train.csv")
train_df = pd.read_csv(train_csv_path)

# Convert file paths for images
train_df["file_name"] = train_df["file_name"].apply(lambda x: os.path.join(dataset_path, x)) # Convert file paths to full paths for images so that they can be read by ImageDataGenerator without any issues
print(train_df.head(20))

# Split dataset into training and validation sets (95% training , 5% validation)
train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=524, stratify=train_df["label"])
print(f"Training set: {train_df.shape}")
print(f"Validation set: {val_df.shape}")

# Convert labels to string format for ImageDataGenerator because it expects labels to be in string format
train_df["label"] = train_df["label"].astype(str)
val_df["label"] = val_df["label"].astype(str)

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data augmentation for training images for generalization
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values
    horizontal_flip=True,  # Flip images horizontally
    rotation_range=20,  # Rotate images randomly
    width_shift_range=0.2,  # Shift image width randomly
    height_shift_range=0.2,  # Shift image height randomly
    zoom_range=0.2  # Apply zoom augmentation
)

# Only rescale validation images with no augmentation because we don't want to validate on augmented images to get a true evaluation of the model
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create data generators for training and validation in order to load images in batches so that the entire dataset does not need to be loaded into memory
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="file_name",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"  # Since it's a binary classification problem with 2 classes real or AI 
) 

# same thing as above for validation
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="file_name",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Define CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)), # Convolutional layer with 32 filters, 3x3 kernel size, ReLU activation function, and same(input size = output size) padding
    BatchNormalization(), # Batch normalization layer to normalize activations so that the model trains faster and is more stable
    MaxPooling2D(pool_size=(2, 2)), # Max pooling layer with 2x2 pool size to reduce spatial dimensions by taking the maximum value in each region
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(), # Dense layers expect a 1D input so we have to flatten the 2D output to 1DS
    Dense(512, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile model with Adam optimizer and binary cross-entropy loss for binary classification
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Display model summary
model.summary()

# Define callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) # Stops training if validation loss does not improve for 3 consecutive epochs
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2) # Reduces learning rate by 0.7 if validation loss does not improve for 2 consecutive epochs

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=[early_stopping, reduce_lr]
)

# Plot training vs validation accuracy
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# Plot training vs validation loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# Save the trained model
model.save("real_or_ai_classifier.h5")
print("Model saved as real_or_ai_classifier.h5")
