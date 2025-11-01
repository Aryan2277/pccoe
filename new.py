from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        api_key = data.get('apiKey')
        model_id = data.get('modelId')
        model_version = data.get('modelVersion')
        image_base64 = data.get('imageBase64')
        confidence = data.get('confidence', 40)
        
        # Roboflow API endpoint
        url = f"https://detect.roboflow.com/{model_id}/{model_version}"
        params = {
            'api_key': api_key,
            'confidence': confidence
        }
        
        # Make request to Roboflow
        response = requests.post(
            url,
            params=params,
            data=image_base64,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)