from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
import io
from segment_anything_hq import sam_model_registry, SamPredictor

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables to store the uploaded image and model
image_np = None
predictor = None

# Load the model once at startup
checkpoint_path = "./pretrained_checkpoint/sam_hq_vit_l.pth"
model_type = "vit_l"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
predictor = SamPredictor(sam)

def preprocess_image(image):
    """
    Preprocess the PIL Image:
    - Convert to NumPy array.
    - Normalize to [0, 1].
    - Convert to PyTorch tensor in [C, H, W] format.
    """
    image_np = np.array(image)

    # Check if the image has the correct number of channels
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("Image must have 3 channels (RGB).")

    # Normalize the image to [0, 1] and convert to float32
    image_np = image_np.astype(np.float32) / 255.0

    # Convert to CHW format
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

    return image_tensor

def run_image_prediction(image_np, predictor, points, point_labels):
    """
    Run image prediction using SAM HQ:
    - Set the image for the predictor.
    - Predict the mask based on the input points.

    Args:
        image_np (np.ndarray): Preprocessed image in [H, W, C] format.
        predictor (SamPredictor): Preloaded SamPredictor object.
        points (list): List of (x, y) coordinates for the mask generation.
        point_labels (list): List of labels for the points (1 for foreground, 0 for background).
    """
    # Convert points and labels to NumPy arrays
    point_coords = np.array(points)
    point_labels = np.array(point_labels)

    # Perform prediction
    with torch.inference_mode():
        try:
            predictor.set_image(image_np)
        except NotImplementedError as e:
            print(f"Error in set_image: {e}")
            return None

        try:
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels
            )
        except AssertionError as e:
            print(f"AssertionError during predict: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during predict: {e}")
            return None

    return masks[0]

@app.route('/upload', methods=['POST'])
def upload_image():
    global image_np

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    try:
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {e}"}), 400

    try:
        image_tensor = preprocess_image(image)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Error preprocessing image: {e}"}), 500

    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    return jsonify({"message": "Image uploaded successfully"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    global image_np, predictor

    if image_np is None:
        return jsonify({"error": "No image uploaded"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    points = data.get('points', None)
    if points is None:
        return jsonify({"error": "No points provided"}), 400

    if not isinstance(points, list) or not all(isinstance(pt, list) and len(pt) == 2 for pt in points):
        return jsonify({"error": "Invalid points format"}), 400

    point_labels = [1] * len(points)  # Assuming all points are foreground

    # Run prediction
    mask = run_image_prediction(image_np, predictor, points, point_labels)

    if mask is None:
        return jsonify({"error": "Prediction failed"}), 500

    # Convert mask to a list for JSON serialization and ensure it's a binary mask
    mask_binary = (mask > 0).astype(int)
    mask_list = mask_binary.tolist()

    return jsonify({"mask": mask_list})

if __name__ == "__main__":
    app.run(debug=True)