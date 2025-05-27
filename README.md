# palcode_assignment

Internship assignment submission for Palcode.ai.

## 🔧 Project Overview

This repository contains the full solution to the object detection assignment provided by Palcode.ai, involving detection of doors and windows in a architecture diagram using a YOLO11 nano model. It includes:

- Trained YOLO11n model
- Inference API using FastAPI
- Training visualizations
- Screenshots and dataset

The deployed app is live and can be tested here:

👉 **[Public Demo on Hugging Face Spaces](https://manthasaigopal-palcode-doors-windows.hf.space/docs)**


## 📁 Directory Structure
```
palcode_assignment/
├── Dataset/           # Contains training and validation images and label files
├── Loss Plots/        # Visualizations of model training loss/metrics
├── Screenshots/       # UI and output screenshots for demonstration
├── doors_windows.pt   # Trained YOLO11n model weights
├── main.py            # FastAPI app for model inference
└── README.md          # This file
```

## 🔄 Example: Testing with curl

You can test the API endpoint by uploading an image file using curl:
```
curl -X POST \
  -F "file=@path_to_your_image.jpg" \
  https://manthasaigopal-palcode-doors-windows.hf.space/detect
```
Replace path_to_your_image.jpg with the actual path to the image you want to test.

## 📷 Screenshots

Refer to the Screenshots/ folder for demonstration images and outputs from the deployed app.

## 📈 Training Artifacts

Visual loss curves and metrics during training are available in the Loss Plots/ directory.
