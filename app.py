#Model code

# from inference_sdk import InferenceHTTPClient
# from PIL import Image
# import requests
# from io import BytesIO

# # Initialize the client for TB detection model
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="weUzBKzyD6TYzQL04eBi"
# )

# def is_xray_image(image_path):
#     # A simple heuristic to check if the image is an X-ray
#     # You could expand this function by using a separate model to detect X-rays
#     try:
#         image = Image.open(image_path)
#         # Check image mode or other basic image properties (e.g., contrast, structure)
#         if image.mode in ['L', 'RGB']:  # X-rays typically use grayscale or RGB for contrast
#             return True
#         else:
#             return False
#     except:
#         return False

# def process_image(image_path):
#     # Step 1: Validate if the image is an X-ray
#     if not is_xray_image(image_path):
#         return {"error": "Not a valid X-ray image. Please upload an X-ray."}
    
#     # Step 2: Perform inference on X-ray image
#     result = CLIENT.infer(image_path, model_id="tb-detection-pigkb/1")
    
#     # Step 3: Return inference result
#     return result

# # Example usage
# image_path = "p.PNG"  # Path to your image

# # Process the image and print result
# result = process_image(image_path)
# print(result)

# # Extract the predictions
# predictions = result.get('predictions', [])

# # Loop through the predictions and display the class and confidence
# for prediction in predictions:
#     detected_class = prediction.get('class', 'Unknown Class')
#     confidence = prediction.get('confidence', 0)

#     print(f"Class: {detected_class}")
#     print(f"Confidence: {confidence:.4f}")
#     print("-" * 30)
    
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Initialize the TB detection model client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="weUzBKzyD6TYzQL04eBi"
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

@app.route('/process-image', methods=['POST'])
def process_image():
    """
    API endpoint to process X-ray images.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']
    
    # Save the uploaded image to a temporary file
    temp_image_path = "temp_image.jpg"
    try:
        image_file.save(temp_image_path)
    except Exception as e:
        print(f"File save error: {e}")
        return jsonify({"error": "Could not save the uploaded image"}), 500

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


# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
