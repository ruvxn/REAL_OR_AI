from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import os
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the CNN model in TFLite format created from to_tflite.py script
TFLITE_MODEL_PATH = "real_or_ai_classifier.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH) 
interpreter.allocate_tensors() # Allocate memory for the model

# Get input and output details of the model 
input_details = interpreter.get_input_details() 
output_details = interpreter.get_output_details()

# Image preprocessing function to match the specified input shape of the model
def preprocess_image(image_path):
    IMG_SIZE = (224, 224)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0  # Normalize pixel values to be between 0 and 1
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Ensure correct datatype for the image
    return image

# Prediction function using TFLite 
def predict(image):
    interpreter.set_tensor(input_details[0]['index'], image) 
    interpreter.invoke() # Run the model
    output_data = interpreter.get_tensor(output_details[0]['index']) 
    return output_data[0][0]  # Extract the prediction value from the output data


# Flask app routes for the web interface
@app.route("/", methods=["GET", "POST"])
def index():
    # Handle POST request for image upload and prediction
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        file = request.files["file"] # Get the uploaded file
        file_path = os.path.join("uploads", file.filename) 
        file.save(file_path)

        print("File successfully uploaded:", file_path)  # Debugging purposes

        # Process and predict using TFLite
        image = preprocess_image(file_path)
        print("Processed Image Shape:", image.shape)  # Debugging purposes

        prediction = predict(image)
        print("Model Prediction Output:", prediction)  # Debugging purposes
        label = "AI-Generated" if prediction > 0.5 else "Real"

        return jsonify({"label": label})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
