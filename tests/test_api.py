import os
import pytest
from fastapi.testclient import TestClient
from src.api.app import app
import numpy as np
from PIL import Image
import io

client = TestClient(app)

def test_read_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()
    assert "Mango Ripeness Detection API" in response.json()["name"]

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

@pytest.fixture
def test_image():
    """Create a test image"""
    # Create a simple 224x224 RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

def test_predict_endpoint(test_image):
    """Test the prediction endpoint with a test image"""
    files = {"file": ("test.jpg", test_image, "image/jpeg")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "predictions" in data
    assert "top_prediction" in data
    assert len(data["predictions"]) == 3  # 3 classes
    
    # Check prediction structure
    for pred in data["predictions"]:
        assert "class_name" in pred
        assert "confidence" in pred
        assert "class_id" in pred
        assert 0 <= pred["confidence"] <= 1
    
    # Check top prediction is one of the classes
    assert data["top_prediction"]["class_name"] in ["unRipe", "ripe", "overRipe"]

def test_invalid_file_upload():
    """Test the prediction endpoint with invalid file"""
    # Test with no file
    response = client.post("/predict")
    assert response.status_code == 422  # Validation error
    
    # Test with non-image file
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "File must be an image" in response.text
