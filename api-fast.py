from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import io
from segment_anything_hq import sam_model_registry, SamPredictor
import asyncio
import uuid
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store the model
predictor = None

# Load the model once at startup
checkpoint_path = "./pretrained_checkpoint/sam_hq_vit_l.pth"
model_type = "vit_l"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
predictor = SamPredictor(sam)

# Dictionary to store images by UUID with timestamps
images = {}
TTL = timedelta(minutes=8)  # Time-to-live for images

class Points(BaseModel):
    points: list[list[float]]
    image_id: str

def preprocess_image(image):
    image_np = np.array(image)

    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("Image must have 3 channels (RGB).")

    image_np = image_np.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

    return image_tensor

async def run_image_prediction(image_np, predictor, points, point_labels):
    point_coords = np.array(points)
    point_labels = np.array(point_labels)

    with torch.inference_mode():
        try:
            predictor.set_image(image_np)
        except NotImplementedError as e:
            logger.error(f"Error in set_image: {e}")
            return None

        try:
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels
            )
        except AssertionError as e:
            logger.error(f"AssertionError during predict: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during predict: {e}")
            return None

    return masks[0]

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        image_tensor = preprocess_image(image)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing image: {e}")

    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_id = str(uuid.uuid4())
    images[image_id] = {"image": image_np, "timestamp": datetime.utcnow()}

    logger.info(f"Image uploaded successfully: {image_id}")

    return JSONResponse(content={"message": "Image uploaded successfully", "image_id": image_id})

@app.post("/predict")
async def predict(data: Points):
    image_id = data.image_id
    points = data.points

    if image_id not in images:
        raise HTTPException(status_code=400, detail="Invalid image ID")

    image_data = images[image_id]
    image_np = image_data["image"]

    if not points:
        raise HTTPException(status_code=400, detail="No points provided")

    if not all(isinstance(pt, list) and len(pt) == 2 for pt in points):
        raise HTTPException(status_code=400, detail="Invalid points format")

    point_labels = [1] * len(points)

    mask = await run_image_prediction(image_np, predictor, points, point_labels)

    if mask is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    mask_binary = (mask > 0).astype(int)
    mask_list = mask_binary.tolist()

    logger.info(f"Prediction successful for image ID: {image_id}")

    return JSONResponse(content={"mask": mask_list})

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_expired_images())

async def cleanup_expired_images():
    while True:
        now = datetime.utcnow()
        expired_keys = [key for key, value in images.items() if now - value["timestamp"] > TTL]
        for key in expired_keys:
            del images[key]
            logger.info(f"Image removed due to TTL expiration: {key}")
        await asyncio.sleep(60)  # Run cleanup every 60 seconds

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)