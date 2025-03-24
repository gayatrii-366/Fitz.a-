from flask import Flask, render_template, request
import cv2
import dlib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load Dlib model
predictor_path = r"c:\Users\Pravin\Downloads\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load dataset for recommendations
fashion_data = pd.read_csv("fashion_recommendations.csv")

def detect_face_shape(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return "No face detected"
    
    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(68)])

        jaw_width = np.linalg.norm(points[0] - points[16])
        cheekbone_width = np.linalg.norm(points[2] - points[14])
        forehead_width = np.linalg.norm(points[19] - points[24])
        face_height = np.linalg.norm(points[8] - points[27])

        if jaw_width < cheekbone_width and forehead_width < cheekbone_width:
            return "Oval"
        elif jaw_width > cheekbone_width:
            return "Square"
        elif forehead_width > cheekbone_width:
            return "Heart"
        else:
            return "Round"

def get_fashion_recommendations(body_shape):
    recommendations = fashion_data[fashion_data["Body Shape"] == body_shape]["Recommendation"].tolist()
    return recommendations if recommendations else ["No recommendations available"]

@app.route("/", methods=["GET", "POST"])
def index():
    face_shape = None
    recommendations = []
    
    if request.method == "POST":
        image_file = request.files["image"]
        image_path = "static/uploads/image.jpg"
        image_file.save(image_path)
        
        face_shape = detect_face_shape(image_path)
        recommendations = get_fashion_recommendations(face_shape)

    return render_template("index.html", face_shape=face_shape, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
