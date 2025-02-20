import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load model trained prevbiously on real_or_ai.py
model = tf.keras.models.load_model("real_or_ai_classifier.h5")

# Define image size suitavle for the model
IMG_SIZE = (224, 224)

# Load the AI generated or real image
image_path = "DALL·E 2025-02-19 23.58.29 - A mesmerizing mountain landscape at sunrise, with snow-capped peaks glowing in the golden light. A crystal-clear river winds through the valley, refle.webp"  
image = cv2.imread(image_path)  # Read the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB 
image = cv2.resize(image, IMG_SIZE)  # Resize to above mentioned image size
image = image / 255.0  # Normalize pixel values to be between 0 and 1
image = np.expand_dims(image, axis=0)  # Add batch dimension so that model can predict by taking batch of images

# Make prediction
# Make prediction
prediction = model.predict(image)

# Interpret result 0 = Real  1 = AI-Generated) based on a threshold
threshold = 0.5  # Since it's a binary classification (sigmoid activation)
label = "AI-Generated" if prediction[0][0] > threshold else "Real"

# Display the label above the image
plt.figure(figsize=(5, 6))
plt.text(0.5, 1.1, label, fontsize=14, ha='center', va='top', fontweight='bold', transform=plt.gca().transAxes)
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()


