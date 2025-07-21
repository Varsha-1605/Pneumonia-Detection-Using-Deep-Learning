import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG19
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2

# ==================== Load Model ====================
base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
dropout = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(dropout)
output = Dense(2, activation='softmax')(class_2)
model_01 = Model(base_model.inputs, output)
# model_01.load_weights('vgg19_model_01.h5')
model_01.load_weights('best_overall_model.h5')
print("✅ Model Loaded Successfully")

# ==================== Flask App ====================
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================== Helper Functions ====================
def getClass(classNo):
    return "Normal" if classNo == 0 else "Pneumonia"

def getResult(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"❌ Could not read image: {img_path}")
    image = Image.fromarray(image).resize((128, 128))
    image = np.array(image) / 255.0
    input_img = np.expand_dims(image, axis=0)
    prediction = model_01.predict(input_img)
    return np.argmax(prediction, axis=1)

# ==================== Routes ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if not file:
            return render_template('index.html', prediction="❌ No file selected.")
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(filepath)

        result_class = getResult(filepath)
        result_text = getClass(result_class[0])

        return render_template('index.html', prediction=f"Prediction: {result_text}")
    except Exception as e:
        return render_template('index.html', prediction=f"❌ Error: {str(e)}")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
