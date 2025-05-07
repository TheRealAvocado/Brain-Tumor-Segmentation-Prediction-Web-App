# 🧠 Brain Tumor Segmentation Web App

This project is a deep learning web application that segments brain tumors from MRI images. It provides a user-friendly frontend for uploading brain scans and a FastAPI-powered backend that runs a trained AI model to predict and visualize tumor regions.

---

## 🚀 Features

- Upload one or multiple brain MRI images (`.png`)
- Real-time prediction using a trained segmentation model
- Highlights tumor areas on the original scan using overlays
- Displays whether cancer is detected
- Saves result images locally for inspection
- Frontend and backend are connected using FastAPI and JavaScript (fetch API)

---

## 🛠️ Technologies Used

- **FastAPI** – for building the backend server
- **TensorFlow / Keras** – for loading and using the segmentation model
- **OpenCV** – for image overlay and processing
- **JavaScript** – for frontend logic and communication
- **HTML/CSS** – for the web interface
- **Uvicorn** – ASGI server to run FastAPI

---

## 📁 Project Structure
 ```text
project/
│
├── main.py           # FastAPI backend server
├── model.pkl         # Trained segmentation model (pickle file)
├── index.html        # Frontend web interface
├── overlay_*.png     # Generated overlay images (after upload)
└── README.md         # Project description and instructions
```



## ⚙️ How to Run the Project

1. **Install the required Python packages:**

   ```bash
   pip install fastapi uvicorn pillow opencv-python tensorflow
2. ** Run the fast api server using this command inside VS Code **
   ```
   uvicorn main:app --reload
   ```
## Results 
![Result Image](overlay_1.png)
