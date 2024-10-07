from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the pretrained VGG19 model
model = VGG19(weights='imagenet')

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image_class(img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0][0]
    return decoded_predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Save the uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Predict the class
        prediction = predict_image_class(filepath)
        label, score = prediction[1], prediction[2]
        
        return render_template('index.html', label=label, score=score)

    return render_template('index.html')

# Function to process all images in a directory
def process_images_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(directory, filename)
            prediction = predict_image_class(img_path)

            # Display the image
            img = image.load_img(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Prediction: {prediction[1]} (Score: {prediction[2]:.2f})")
            plt.show()

if __name__ == '__main__':
    app.run(debug=True)
