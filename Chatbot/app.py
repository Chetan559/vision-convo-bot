from flask import Flask, request, jsonify
from flask_cors import CORS
from chat import get_response 
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)  
CORS(app)  

@app.post("/answer")  
def predict():
    text = request.get_json().get("message")  # Get the "message" from the request's JSON data
    response = get_response(text)  # Get a response using the 'get_response' function from chat.py
    message = {"answer": response }  # Createing a dictionary with the response
    return jsonify(message)  # Returning the response as a JSON object


if __name__ == '__main__':
    app.run(debug=True, port=5500) 
