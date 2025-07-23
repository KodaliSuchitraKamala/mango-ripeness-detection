import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import tensorflow as tf
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Import model and preprocessing
from src.models.model import MangoRipenessModel

# Initialize FastAPI app
app = FastAPI(
    title="Mango Ripeness Detection API",
    description="API for detecting mango ripeness using deep learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and configuration
MODEL_PATH = "models/mango_ripeness_model.h5"  # Update this path as needed
CLASS_NAMES = ["overRipe", "ripe", "unRipe"]
IMG_SIZE = (224, 224)

# Load the model
model = None

def load_model():
    """Load the trained model."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load model on startup
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Request/Response models
class PredictionResult(BaseModel):
    class_name: str
    confidence: float
    class_id: int

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    top_prediction: PredictionResult

# Helper functions
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess the image for prediction."""
    # Resize and normalize the image
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype('float32') / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# API endpoints
@app.get("/")
async def read_root():
    """Root endpoint with API information."""
    return {
        "name": "Mango Ripeness Detection API",
        "version": "1.0.0",
        "description": "API for detecting mango ripeness using deep learning",
        "endpoints": {
            "/predict": "POST - Upload an image of a mango to detect its ripeness",
            "/health": "GET - Check if the API is running"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict the ripeness of a mango from an uploaded image.
    
    Args:
        file: Image file to predict on (JPG, PNG, etc.)
        
    Returns:
        Prediction results including class probabilities
    """
    # Check if file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)[0]
        
        # Get top prediction
        class_id = np.argmax(predictions)
        confidence = float(predictions[class_id])
        
        # Prepare response
        results = []
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
            results.append({
                "class_name": class_name,
                "confidence": float(prob),
                "class_id": i
            })
        
        # Sort by confidence in descending order
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "predictions": results,
            "top_prediction": {
                "class_name": CLASS_NAMES[class_id],
                "confidence": confidence,
                "class_id": int(class_id)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
