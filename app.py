import os
import gdown
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Google Drive File ID & Model Path
FILE_ID = "1f3NEKmnPa6slddXjLYgKWXtMRZEDg34s"
MODEL_PATH = "CNN_TRAINED_MODEL.tflite"

# Function to download the model if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        try:
            gdown.download(URL, MODEL_PATH, quiet=False)
            if not os.path.exists(MODEL_PATH):
                raise Exception("❌ Model download failed!")
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"⚠️ Error downloading model: {e}")
            exit(1)

# Download the model before proceeding
download_model()

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_labels = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)_healthy', 'Cherry(including_sour)_Powdery_mildew',
    'Corn(maize)_Cercospora_leaf_spot_Gray_leaf_spot', 'Corn(maize)_Common_rust', 'Corn(maize)_healthy',
    'Corn(maize)_Northern_Leaf_Blight', 'Grape_Black_rot', 'Grape__Esca(Black_Measles)', 
    'Grape__healthy', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Orange__Haunglongbing(Citrus_greening)', 
    'Peach__Bacterial_spot', 'Peach_healthy', 'Pepper_bell__Bacterial_spot', 'Pepper_bell__healthy', 
    'Potato__Early_blight', 'Potato_healthy', 'Potato_Late_blight', 'Raspberry__healthy', 
    'Soybean__healthy', 'Squash__Powdery_mildew', 'Strawberry_healthy', 'Strawberry__Leaf_scorch', 
    'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato__Late_blight', 
    'Tomato__Leaf_Mold', 'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites_Two-spotted_spider_mite', 
    'Tomato__Target_Spot', 'Tomato_Tomato_mosaic_virus', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus'
]

# Disease details dictionary
disease_details = {
    'Apple__Apple_scab': "Apply fungicides labeled for apple scab. Prune and remove infected leaves.",
    'Apple_Black_rot': "Use fungicides and remove infected plant material.",
    'Apple_Cedar_apple_rust': "Use fungicide treatments and consider removing nearby cedar trees.",
    'Apple__healthy': "No treatment required.",
    'Blueberry__healthy': "No treatment required.",
    'Cherry(including_sour)_healthy': "No treatment required.",
    'Cherry(including_sour)_Powdery_mildew': "Apply fungicides and maintain air circulation.",
    'Corn(maize)_Cercospora_leaf_spot_Gray_leaf_spot': "Rotate crops and use fungicides.",
    'Corn(maize)_Common_rust': "Use resistant varieties and apply fungicides if necessary.",
    'Corn(maize)_healthy': "No treatment required.",
    'Corn(maize)_Northern_Leaf_Blight': "Rotate crops and use resistant varieties.",
    'Grape_Black_rot': "Apply fungicides and remove infected plant parts.",
    'Grape__Esca(Black_Measles)': "Prune infected vines and apply fungicides.",
    'Grape__healthy': "No treatment required.",
    'Grape_Leaf_blight(Isariopsis_Leaf_Spot)': "Apply fungicides for grape diseases.",
    'Orange__Haunglongbing(Citrus_greening)': "No cure. Remove infected trees.",
    'Peach__Bacterial_spot': "Use copper-based sprays and prune infected branches.",
    'Peach_healthy': "No treatment required.",
    'Pepper_bell__Bacterial_spot': "Use copper-based sprays and avoid overhead watering.",
    'Pepper_bell__healthy': "No treatment required.",
    'Potato__Early_blight': "Apply fungicides and rotate crops.",
    'Potato_healthy': "No treatment required.",
    'Potato_Late_blight': "Apply fungicides and avoid overhead watering.",
    'Raspberry__healthy': "No treatment required.",
    'Soybean__healthy': "No treatment required.", 'Squash__Powdery_mildew': "Use fungicides and improve air circulation.",
    'Strawberry_healthy': "No treatment required.",
    'Strawberry__Leaf_scorch': "Improve air circulation and remove infected leaves.",
    'Tomato__Bacterial_spot': "Use copper-based sprays and avoid overhead watering.",
    'Tomato_Early_blight': "Apply fungicides and rotate crops.",
    'Tomato_healthy': "No treatment required.",
    'Tomato__Late_blight': "Apply fungicides and avoid overhead watering.",
    'Tomato__Leaf_Mold': "Apply fungicides and maintain air circulation.",
    'Tomato__Septoria_leaf_spot': "Use fungicides for tomato diseases.",
    'Tomato__Spider_mites_Two-spotted_spider_mite': "Use miticides and keep the growing area clean.",
    'Tomato__Target_Spot': "Apply fungicides for tomato diseases.",
    'Tomato_Tomato_mosaic_virus': "No cure. Remove infected plants.",
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus': "Use resistant varieties and control whiteflies.",
}


# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    image = np.array(image).astype('float32') / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Step 2: Add Homepage Route
@app.route('/')
def home():
    return "Welcome to the Plant Disease Detection API! Use the /predict endpoint to analyze plant diseases."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess image
        input_image = preprocess_image(image)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get prediction
        predicted_class_index = np.argmax(output_data, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]

        # Get disease details
        disease_info = disease_details.get(predicted_class_label, "No details available.")

        # Return prediction and details as JSON
        return jsonify({
            "prediction": predicted_class_label,
            "disease_details": disease_info
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
