from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

model = YOLO("doors_windows.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model(image)
    result = results[0]

    detections = []
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        bbox = [x1, y1, x2 - x1, y2 - y1]

        detections.append({
            "label": label,
            "confidence": round(conf, 2),
            "bbox": [round(v, 2) for v in bbox]
        })

    return {"detections": detections}

@app.get("/")
def read_root():
    return {"message": "API is running. Use /docs for Swagger UI."}