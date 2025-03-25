import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# Load recommendation dataset
df = pd.read_csv("C:\\Users\\Pravin\\fitz-ai\\fashion_recommendations.csv")

app = Flask(__name__)

# Face shape detection function
def detect_face_shape(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "Unknown"

    for (x, y, w, h) in faces:
        aspect_ratio = w / h
        if aspect_ratio > 1.1:
            return "Round"
        elif aspect_ratio < 0.9:
            return "Oval"
        else:
            return "Square"
    
    return "Unknown"

# Skin tone analysis based on image
def detect_skin_tone(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    avg_color = np.mean(image, axis=(0, 1))

    r, g, b = avg_color
    if r > g and r > b:
        return "Warm"
    elif b > r and b > g:
        return "Cool"
    else:
        return "Neutral"

# Color recommendation based on skin tone
def recommend_colors(skin_tone):
    warm_colors = ["Red", "Orange", "Yellow", "Brown"]
    cool_colors = ["Blue", "Green", "Purple"]
    neutral_colors = ["Black", "White", "Gray"]

    if skin_tone == "Warm":
        return warm_colors
    elif skin_tone == "Cool":
        return cool_colors
    else:
        return neutral_colors

# Determine body shape based on measurements
def determine_body_shape(shoulders, bust, waist, hips):
    if abs(shoulders - hips) <= 1 and abs(bust - hips) <= 1:
        return "Hourglass"
    elif hips > shoulders and hips > bust:
        return "Pear"
    elif shoulders > hips and shoulders > bust:
        return "Inverted Triangle"
    elif waist < bust and waist < hips:
        return "Apple"
    else:
        return "Rectangle"

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if request.method == "POST":
        shoulders = float(request.form["shoulders"])
        bust = float(request.form["bust"])
        waist = float(request.form["waist"])
        hips = float(request.form["hips"])
        body_shape = determine_body_shape(shoulders, bust, waist, hips)
        return render_template("results.html", body_shape=body_shape)
    return render_template("quix.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    image = request.files["image"]
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    face_shape = detect_face_shape(image_path)
    skin_tone = detect_skin_tone(image_path)
    color_suggestions = recommend_colors(skin_tone)

    return render_template("results.html", face_shape=face_shape, skin_tone=skin_tone, color_suggestions=color_suggestions)

if __name__ == "__main__":
    app.run(debug=True)
