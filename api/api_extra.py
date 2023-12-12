from flask import Flask, request, jsonify
from fastai.vision.all import *
from PIL import Image

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the fastai learner model
model_path = "./models/fastai_model.pkl"
learn = load_learner(model_path)

def predict_image(img_path):
    # Open and preprocess the image
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((224, 224))
    img_fastai = PILImage.create(img_resized)

    # Perform inference using the loaded model
    pred_class, pred_idx, probabilities = learn.predict(img_fastai)

    return {"Predicted Class": str(pred_class), "Prediction Probability": float(probabilities[pred_idx])}


@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return jsonify({"error": "No file attached"}), 400

    file = request.files['file']

    # Check if the file is allowed
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file format"}), 400

    # Save the uploaded file
    uploaded_file_path = "uploaded_image.jpg"
    file.save(uploaded_file_path)

    # Make prediction
    prediction = predict_image(uploaded_file_path)

    # Remove the uploaded file
    os.remove(uploaded_file_path)

    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=False)