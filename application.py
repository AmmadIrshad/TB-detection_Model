from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os
import uuid
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)

# Initialize the TB detection model client
load_dotenv()
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")  # Use environment variable for API key
)

def is_xray_image(file_path):
    """
    Check if the image appears to be an X-ray image.
    """
    try:
        img = Image.open(file_path)
        if img.mode in ['L', 'RGB']:  # Valid image modes for X-rays
            return True
    except Exception as e:
        print(f"Image validation error: {e}")
    return False

@app.route('/', methods=['GET'])
def welcome():
    """
    Root endpoint that returns a welcome message.
    """
    return jsonify({"message": "Welcome to the Tuberculosis API! Go to /process-image endpoint for detection."}), 200

@app.route('/process-image', methods=['POST'])
def process_image():
    """
    API endpoint to process X-ray images.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']

    if not image_file or image_file.filename == '':
        return jsonify({"error": "Invalid file uploaded"}), 400

    # Use a unique file name in the current working directory
    temp_image_path = os.path.join(os.getcwd(), f"temp_{uuid.uuid4().hex}.jpg")

    try:
        print(f"Saving file to: {temp_image_path}")
        image_file.save(temp_image_path)
        print(f"File saved successfully to: {temp_image_path}")
    except Exception as e:
        print(f"File save error: {e}")
        return jsonify({"error": f"Could not save the uploaded image: {str(e)}"}), 500

    # Check if the saved file is a valid X-ray image
    if not is_xray_image(temp_image_path):
        os.remove(temp_image_path)  # Clean up
        return jsonify({"error": "Uploaded file is not a valid X-ray image"}), 400

    try:
        # Perform inference using the model
        result = CLIENT.infer(temp_image_path, model_id="tb-detection-pigkb/1")

        # Clean up temporary file after inference
        os.remove(temp_image_path)

        # Return inference result
        return jsonify(result), 200

    except Exception as e:
        # Handle inference errors
        print(f"Inference error: {e}")
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)  # Clean up
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))