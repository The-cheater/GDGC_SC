from flask import Flask, request, jsonify, render_template
import uuid
import os
from adhaar import detect_features
from flask_cors import CORS
CORS(app)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5MB limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('aadhaar.html')

@app.route('/verify', methods=['POST'])
def verify():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "Empty file"}), 400
    
    # Save the uploaded image temporarily
    filename = str(uuid.uuid4()) + '.png'
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(save_path)
    
    try:
        # Process with your ML model
        result, _ = detect_features(save_path)
        return jsonify({"verified": bool(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup
        if os.path.exists(save_path):
            os.remove(save_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)