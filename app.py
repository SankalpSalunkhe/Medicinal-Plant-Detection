#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify


# In[ ]:


# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# In[ ]:


# Dataset Path
data_dir = r"Medicinal plant dataset"
img_size = (224, 224)
batch_size = 8


# In[ ]:


# Load Pretrained Model Instead of Training Again
model_path = "best_model.h5"

if os.path.exists(model_path):
    model = keras.models.load_model(model_path)  # ✅ Load trained model
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"🚨 Model file '{model_path}' not found!")


# In[ ]:


# Plant Descriptions
descriptions = {
    "Aloevera": "Known for its soothing gel, used to treat burns, wounds, and skin irritations.",
    "Amla": "Rich in vitamin C, boosts immunity, and supports digestion.",
    "Amruta_balli": "Used in Ayurveda for detoxification and treating diabetes.",
    "Arali": "Traditionally used for skin infections and wound healing.",
    "Ashoka": "Used in Ayurvedic medicine to treat menstrual disorders and improve skin health.",
    "Ashwagandha": "Known as an adaptogen, it helps reduce stress and improve energy levels.",
    "Avocado": "High in healthy fats, good for heart health and skin nourishment.",
    "Bamboo": "Used in herbal remedies for joint pain and as an antioxidant.",
    "Basale": "Rich in iron and vitamin C, used for healing wounds and treating anemia.",
    "Betel": "Chewing betel leaves improves digestion and has antibacterial properties.",
    "Betel_nut": "Used in traditional medicine for digestion and as a stimulant.",
    "Brahmi": "Improves memory, cognitive function, and reduces anxiety.",
    "Castor": "Used to make castor oil, which treats constipation and skin conditions.",
    "Curry_leaf": "Enhances digestion, improves eyesight, and controls diabetes.",
    "Doddapatre": "Used for treating colds, coughs, and digestive issues.",
    "Ekka": "Used in traditional medicine for treating fever and respiratory issues.",
    "Ganike": "Known for its anti-inflammatory properties and used for wound healing.",
    "Gauva": "Rich in vitamin C, aids digestion, and boosts immunity.",
    "Geranium": "Used in aromatherapy for stress relief and skin care.",
    "Henna": "Known for its cooling effect, used as a natural dye and for scalp health.",
    "Hibiscus": "Supports heart health, lowers blood pressure, and promotes hair growth.",
    "Honge": "Used for oil extraction with medicinal benefits for joint pain.",
    "Insulin": "Used in herbal medicine for managing diabetes and blood sugar levels.",
    "Jasmine": "Known for its calming properties, used in aromatherapy and skincare.",
    "Lemon": "Rich in vitamin C, aids digestion, and detoxifies the body.",
    "Lemon_grass": "Used in tea for relaxation and improving digestion.",
    "Mango": "Rich in antioxidants, promotes digestion, and supports skin health.",
    "Mint": "Used for digestion, fresh breath, and cooling effects.",
    "Nagadali": "Traditional medicinal plant used for treating fever and skin disorders.",
    "Neem": "Known for its antibacterial properties, used for skin care and oral health.",
    "Nithyapushpa": "Used for wound healing and anti-inflammatory benefits.",
    "Nooni": "Supports immune health and is rich in antioxidants.",
    "Pappaya": "Supports digestion with its enzyme papain, good for skin health.",
    "Pepper": "Known for improving digestion and boosting metabolism.",
    "Pomegranate": "Rich in antioxidants, supports heart health and skin glow.",
    "Raktachandini": "Used in Ayurveda for treating skin diseases and detoxification.",
    "Rose": "Used in skincare, aromatherapy, and for improving digestion.",
    "Sapota": "High in fiber and good for digestion and immunity.",
    "Tulasi": "Known for boosting immunity, treating colds, and respiratory issues.",
    "Wood_sorel": "Used for its cooling effect and to treat urinary infections."
}


# In[ ]:


# Data Augmentation & Preprocessing
#datagen = ImageDataGenerator(
#    rescale=1./255,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    horizontal_flip=True,
#    validation_split=0.2  # 80% Train, 20% Validation
#)


# In[ ]:


# Train & Validation Data Loaders
#train_generator = datagen.flow_from_directory(
#    data_dir,
#    target_size=img_size,
#    batch_size=batch_size,
#    class_mode="sparse",
#    subset="training"
#)

#val_generator = datagen.flow_from_directory(
#    data_dir,
#    target_size=img_size,
#    batch_size=batch_size,
#    class_mode="sparse",
#    subset="validation"
#)


# In[ ]:


# Load Pretrained Model
#base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[ ]:


# Unfreeze Last Few Layers
#for layer in base_model.layers[-10:]:  # Unfreezing last 10 layers
#   layer.trainable = True


# In[ ]:


# Build Model
#model = models.Sequential([
#    base_model,
#    layers.GlobalAveragePooling2D(),
#    layers.Dense(128, activation='relu'),
#    layers.Dropout(0.3),
#    layers.Dense(train_generator.num_classes, activation='softmax')
#])


# In[ ]:


# Compile Model with Slightly Higher Learning Rate
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])


# In[ ]:


# Train Model with Checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True)

#history = model.fit(
#    train_generator,
#    validation_data=val_generator,
#    epochs=15,
#    callbacks=[checkpoint]
#)


# In[ ]:


# Function to Predict from Uploaded Image
def predict_plant(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Class labels (manually mapping based on dataset)
    class_labels = ["Aloevera", "Amla", "Amruta_balli", "Arali", "Ashoka", "Ashwagandha", "Avocado", "Bamboo", "Basale", "Betel", "Betel_nut", "Brahmi", "Castor", "Curry_leaf", "Doddapatre", "Ekka", "Ganike", "Gauva", "Geranium", "Henna", "Hibiscus", "Honge", "Insulin", "Jasmine", "Lemon", "Lemon_grass", "Mango", "Mint", "Nagadali", "Neem", "Nithyapushpa", "Nooni", "Pappaya", "Pepper", "Pomegranate", "Raktachandini", "Rose", "Sapota", "Tulasi", "Wood_sorel"]
    
    plant_name = class_labels[class_index]
    description = descriptions.get(plant_name, "Description not available")
    return plant_name, confidence, description


# In[ ]:


# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    plant, confidence, description = None, None, None  # Fix: Removed extra variable 'filename'
    filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', plant=None)
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', plant=None)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            plant, confidence, description = predict_plant(filepath)
    return render_template('index.html', plant=plant, confidence=confidence, description=description, image=filename)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        plant, confidence, description = predict_plant(filepath)
        return jsonify({
            'plant': plant,
            'confidence': float(confidence),  # Convert to standard Python float
            'description': description
        })

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:


import os
print(os.path.exists("best_model.h5"))


# In[ ]:

