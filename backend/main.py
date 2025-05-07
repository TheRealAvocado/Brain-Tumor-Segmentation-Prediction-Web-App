import os
import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from PIL import Image
import cv2 as cv
import base64
from io import BytesIO
import tensorflow as tf

app = FastAPI()

# Allow CORS for development purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Helper function to create overlay
def create_overlay(img, mask, alpha=0.6):
    """
    Create an overlay of the tumor mask on the MRI image.
    
    Parameters:
        img (numpy array): Original MRI image.
        mask (numpy array): Tumor segmentation mask.
        alpha (float): Transparency level for the overlay.
    
    Returns:
        overlay: The overlaid image.
    """
    # Normalize the image for display
    if img.max() > 1:
        img = (img / img.max()) * 255
    img = img.astype(np.uint8)
    
    # Normalize the mask (binary mask with tumor regions)
    mask = (mask > 0).astype(np.uint8) * 255  # Ensure it's binary (0 or 255)
    
    # Convert grayscale image to BGR for visualization
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
     # Create a red mask to highlight tumor regions
    tumor_highlight = np.zeros_like(img)
    tumor_highlight[:, :, 2] = mask  # Red channel only

    # Blend the original image and the tumor highlight
    overlay = cv.addWeighted(img, 1 - alpha, tumor_highlight, alpha, 0)
    
    return overlay

# Helper function to load the pickled model
def load_pickled_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model

# Prediction helper
def predict_tumor_mask(model, img_bytes, input_size=240):
    test_img = Image.open(BytesIO(img_bytes))
    resized_img = np.array(test_img.resize((input_size, input_size)))

    # Predict tumor mask
    result = model.predict(np.array([resized_img]))[0].transpose(2, 0, 1)[0]

    # Noise removal
    kernel = np.ones((10, 10), np.uint8)
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
    result = np.where(result > 0.8, 1, 0)

    # Determine if the slice has a tumor
    # FASTAPI doesn't work with numpy.bool_ so we change it to bool
    has_cancer = bool(np.any(result > 0))
    tumor_pixel_count = int(np.sum(result))

    return resized_img, result, has_cancer, tumor_pixel_count
    # return resized_img, result, has_cancer

@app.post("/predict-cancer/")
async def predict_cancer(
    files: List[UploadFile] = File(...),  # Accept one or more images
    # Update model path accordingly
    model_path: str = "D:/Study/Fourth Year/HIS/model.pkl"  
):
    # Load the segmentation model
    model = load_pickled_model(model_path)
    # Initialize results
    predictions = []
    overall_cancer_detected = False
    for uploaded_file in files:
        try:
            # Read image bytes
            img_bytes = await uploaded_file.read()
            # Predict tumor segmentation
            img, result, has_cancer,tumor_pixel_count = predict_tumor_mask(model, img_bytes)
            print(f"Result for {uploaded_file.filename}: has_cancer={has_cancer}")
            # Update overall cancer detection status
            # Initialize overlay_filename (default to None)
            overlay_filename = None
            if has_cancer:
                overall_cancer_detected = True
                # Create overlay of the mask on the MRI slice
                overlay = create_overlay(img, result)
                # Save the overlay image for debugging/visualization
                overlay_filename = f"overlay_{uploaded_file.filename}.png"
                cv.imwrite(overlay_filename, overlay)
                print(f"Saved overlay image as {overlay_filename}")
                
            # Add to results
            predictions.append({
                "filename": uploaded_file.filename,
                "has_cancer": has_cancer,
                "tumor_pixel_count": tumor_pixel_count,
                "overlay_image_path": overlay_filename,
            })
                
        except Exception as e:
            # Log errors and skip problematic files
            print(f"Error processing {uploaded_file.filename}: {e}")

    # Prepare the response
    response = {
        "overall_cancer_detected": overall_cancer_detected,
        "predictions": predictions,
    }
    return JSONResponse(content=response)
