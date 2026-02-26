from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io
from src.predict import predict_image

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Cricket Shot Classification API Running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)

    predicted_class, confidence = predict_image(image)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }