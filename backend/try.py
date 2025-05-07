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

# Helper function to load the pickled model
def load_pickled_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        # model.save('model.h5')  # Save it in h5 format

        # Load it back easily using TensorFlow
        # model = tf.keras.models.load_model('model.h5')
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
    model_path: str = "D:/Study/Fourth Year/HIS/model.pkl"  # Update model path accordingly
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
            
            # result = predict_tumor_mask(model, img_bytes)  # Ensure this returns (img, result, has_cancer)
            # if len(result) != 3:
            #     raise ValueError("Unexpected number of return values from predict_tumor_mask")

            # img, mask, has_cancer = result
            
            #download the image with cancer
            cv.imwrite(f"debug_result_{uploaded_file}.png", result * 255)
            
            # Add to results
            predictions.append({
                "filename": uploaded_file.filename,
                "has_cancer": has_cancer,
            })

            # Update overall cancer detection status
            if has_cancer:
                overall_cancer_detected = True

        except Exception as e:
            # Log errors and skip problematic files
            print(f"Error processing {uploaded_file.filename}: {e}")

    # Prepare the response
    response = {
        "overall_cancer_detected": overall_cancer_detected,
        "predictions": predictions,
    }
    return JSONResponse(content=response)
