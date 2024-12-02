# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="weUzBKzyD6TYzQL04eBi"
# )

# result = CLIENT.infer('dog-68-207x300.png', model_id="tb-detection-pigkb/1")
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

from inference_sdk import InferenceHTTPClient
from PIL import Image
import requests
from io import BytesIO

# Initialize the client for TB detection model
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="weUzBKzyD6TYzQL04eBi"
)

def is_xray_image(image_path):
    # A simple heuristic to check if the image is an X-ray
    # You could expand this function by using a separate model to detect X-rays
    try:
        image = Image.open(image_path)
        # Check image mode or other basic image properties (e.g., contrast, structure)
        if image.mode in ['L', 'RGB']:  # X-rays typically use grayscale or RGB for contrast
            return True
        else:
            return False
    except:
        return False

def process_image(image_path):
    # Step 1: Validate if the image is an X-ray
    if not is_xray_image(image_path):
        return {"error": "Not a valid X-ray image. Please upload an X-ray."}
    
    # Step 2: Perform inference on X-ray image
    result = CLIENT.infer(image_path, model_id="tb-detection-pigkb/1")
    
    # Step 3: Return inference result
    return result

# Example usage
image_path = "p.PNG"  # Path to your image

# Process the image and print result
result = process_image(image_path)
print(result)

# Extract the predictions
predictions = result.get('predictions', [])

# Loop through the predictions and display the class and confidence
for prediction in predictions:
    detected_class = prediction.get('class', 'Unknown Class')
    confidence = prediction.get('confidence', 0)

    print(f"Class: {detected_class}")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 30)
    