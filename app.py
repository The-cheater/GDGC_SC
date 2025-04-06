from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './temp'

def validate_aadhaar(image_file):
    """Simulate Aadhaar validation - replace with your actual ML model"""
    try:
        # Read and preprocess image
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Here you would normally use your ML model
        # For testing, alternate between valid and invalid responses
        current_second = datetime.now().second
        is_valid = current_second % 2 == 0  # Alternate True/False every second
        
        return {
            'is_valid': is_valid,
            'confidence': 0.95 if is_valid else 0.45,
            'error': None if is_valid else "Potential fake detected"
        }
    except Exception as e:
        return {
            'is_valid': False,
            'confidence': 0.0,
            'error': str(e)
        }

@app.route('/')
def index():
    return render_template('aadhaar.html')

@app.route('/verify', methods=['POST'])
def verify():
    if 'image' not in request.files:
        return jsonify({'verified': False, 'error': 'No image provided'}), 400
        
    try:
        image = request.files['image']
        result = validate_aadhaar(image)
        
        return jsonify({
            'verified': result['is_valid'],
            'confidence': result['confidence'],
            'error': result.get('error', '')
        })
    except Exception as e:
        return jsonify({'verified': False, 'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)