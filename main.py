import torch
import cv2
import numpy as np
from segment_anything_hq import sam_model_registry, SamPredictor

# Load the model
model_type = "vit_l"
sam_checkpoint = "pretrained_checkpoint/sam_hq_vit_l.pth"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Load the image
image_path = "selection-140.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set the image in the predictor
predictor.set_image(image_rgb)

# Function to handle mouse click
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1 indicates a positive point
        masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label)
        
        # Display the mask on the image
        mask = masks[0]
        mask_image = (mask * 255).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        combined_image = cv2.addWeighted(image, 0.7, mask_image, 0.3, 0)
        
        cv2.imshow("Image with Mask", combined_image)

# Display the image and set the mouse callback
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()