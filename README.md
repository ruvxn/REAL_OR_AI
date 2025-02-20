### **Real vs AI Image Classifier**  

#### **Project Overview**  
This project is a **machine learning-based image classification system** designed to distinguish between **real images and AI-generated images**. The system is built using **TensorFlow** and **OpenCV** for image processing, and it is deployed using **Flask** for a web-based interface.

Users can upload an image through a simple **web interface**, and the model will predict whether the image is **real** or **AI-generated**.

---

### **Dataset**
The dataset used for this project is sourced from **Kaggle**:
```sh
!kaggle datasets download -d alessandrasala79/ai-vs-human-generated-dataset
```
This dataset contains a variety of images, including **real photographs and AI-generated visuals**, covering different categories beyond just human faces.

#### **Data Preprocessing:**
- **Resizing** images to a fixed size (e.g., 128x128 pixels).
- **Normalization** to scale pixel values between 0 and 1.
- **Data Augmentation** to improve model robustness (random flips, rotations, etc.).
- **Splitting** into training, validation, and test sets.

---

### **Features**
- **Convolutional Neural Network (CNN):** The model is built using deep learning to analyze image features.
- **Web-based Interface:** Users can upload images via a **Flask-powered front end**.
- **Real-time Predictions:** The model classifies images as either "Real" or "AI-generated."
- **Lightweight Deployment:** Supports TensorFlow Lite for mobile-friendly execution.

---

### **Project Structure**
- **`app.py`** - Flask backend to handle image uploads and serve predictions.  
- **`real_or_ai.py`** - Main script to load the trained model and classify images.  
- **`to_tflite.py`** - Converts the trained TensorFlow model to TensorFlow Lite format.  
- **`Test.py`** - Script for testing and validating model performance.  
- **`index.html`** - Front-end web interface for uploading images and displaying predictions.  
- **`requirements.txt`** - Lists dependencies required to run the project.

---

### **Installation**
1. **Clone the repository:**
   ```sh
   git clone <repository_url>
   cd real_or_ai
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Flask app:**
   ```sh
   python app.py
   ```

4. **Access the web app:**  
   Open your browser and go to `http://127.0.0.1:5000/`

---

### **Dependencies**
The project requires the following libraries (listed in `requirements.txt`):  
- **TensorFlow** - Deep learning framework  
- **Matplotlib** - Visualization tools  
- **OpenCV** - Image processing  
- **Pandas** - Data handling  
- **Scikit-learn** - Machine learning utilities  
- **tqdm** - Progress bar for training  

Install them using:
```sh
pip install -r requirements.txt
```

---

### **Model Training & Conversion**
- The model is trained using a dataset of real and AI-generated images.
- It is then converted into **TensorFlow Lite** using `to_tflite.py` for efficient deployment.

---

### **How It Works**
1. **Upload an image** via the front-end interface.
2. The Flask backend processes the image and runs the trained CNN model.
3. The result is displayed on the webpage, classifying the image as either **"Real"** or **"AI-generated."**

---

### **Future Improvements**
- Fine-tune model accuracy with more datasets.
- Optimize for **mobile and embedded systems** using TensorFlow Lite.
- Enhance UI/UX with more interactive features.

