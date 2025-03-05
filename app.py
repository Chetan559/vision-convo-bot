from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from yolov8_basics import object_predictor

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is working"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image file is present in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Retrieve the image file from the request
    file = request.files['image']
    # Convert the image file to a numpy array
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Check if the image was correctly decoded
    if image is None:
        return jsonify({"error": "Failed to decode the image"}), 400

    # Perform prediction
    try:
        detection_details = object_predictor(image)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Return the detection details as JSON
    return jsonify(detection_details)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
