import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load fashion dataset (sample structure)
fashion_data = pd.DataFrame({
    'type': ['hairstyle', 'eyewear', 'top', 'dress'],
    'features': [
        ['round', 'layers', 'long'],
        ['angular', 'cat-eye', 'rectangular'],
        ['v-neck', 'wrap', 'peplum'],
        ['a-line', 'bodycon', 'shift']
    ],
    'body_type': ['hourglass', 'pear', 'rectangle', 'apple'],
    'face_shape': ['oval', 'square', 'round', 'heart'],
    'color_palette': ['warm', 'cool', 'neutral', 'neutral']
})

# Use MultiLabelBinarizer to convert features to a one-hot encoded format
mlb = MultiLabelBinarizer()
features_encoded = mlb.fit_transform(fashion_data['features'])

# AI Model for recommendations
nn_model = NearestNeighbors(n_neighbors=5)
nn_model.fit(features_encoded)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def analyze_color(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return "Error loading image"
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Implement color analysis logic here
    return "color_profile_placeholder"

def analyze_face_shape(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return "Error loading image"
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Implement face shape classification logic here
    return "face_shape_placeholder"

def get_ai_recommendations(features):
    features_vector = mlb.transform([features])
    _, indices = nn_model.kneighbors(features_vector)
    return fashion_data.iloc[indices[0]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    # Process image
    color_profile = analyze_color(filename)
    face_shape = analyze_face_shape(filename)
    
    return redirect(url_for('quiz', 
                          color=color_profile,
                          face_shape=face_shape,
                          img_path=filename))

@app.route('/quiz')
def quiz():
    return render_template('quiz.html',
                         color=request.args.get('color'),
                         face_shape=request.args.get('face_shape'),
                         img_path=request.args.get('img_path'))

@app.route('/recommend', methods=['POST'])
def recommend():
    user_data = {
        'color': request.form['color'],
        'face_shape': request.form['face_shape'],
        'body_measurements': {
            'height': float(request.form['height']),
            'weight': float(request.form['weight']),
            'shoulders': request.form['shoulders'],
            'hips': request.form['hips']
        },
        'style_preferences': request.form.getlist('preferences')
    }
    
    features = user_data['style_preferences']
    recommendations = get_ai_recommendations(features)
    
    return render_template('results.html',
                         recommendations=recommendations,
                         user_data=user_data)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)