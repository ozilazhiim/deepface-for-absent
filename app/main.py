import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from deepface import DeepFace
from typing import List
import os

app = FastAPI(title="DeepFace Local API")
os.environ["DEEPFACE_HOME"] = "/app/weights"

print("DeepFace API siap dijalankan...")

async def read_image(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="File bukan gambar valid")
    return img

@app.get("/")
def home():
    return {"message": "DeepFace API is Running. Access /docs for UI."}

# --- 2. ENDPOINT ANALYZE (Umur, Gender, Emosi) ---
@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    try:
        img = await read_image(file)
        obj = DeepFace.analyze(img_path=img, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        
        if isinstance(obj, list):
            result = obj[0]
        else:
            result = obj
            
        return {
            "status": "success",
            "age": result.get('age'),
            "gender": result.get('dominant_gender'),
            "emotion": result.get('dominant_emotion'),
            "box": result.get('region')
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/verify")
async def verify_faces(img1: UploadFile = File(...), img2: UploadFile = File(...)):
    try:
        image1 = await read_image(img1)
        image2 = await read_image(img2)
        result = DeepFace.verify(img1_path=image1, img2_path=image2, model_name="VGG-Face", enforce_detection=False)

        return {
            "verified": result['verified'],
            "distance": result['distance'],
            "threshold": result['threshold'],
            "model": result['model'],
            "similarity_metric": result['similarity_metric']
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/represent")
async def get_embedding(file: UploadFile = File(...)):
    try:
        img = await read_image(file)
        
        embedding_objs = DeepFace.represent(img_path=img, model_name="VGG-Face", enforce_detection=False)
        
        if len(embedding_objs) > 0:
            embedding = embedding_objs[0]["embedding"]
            return {"embedding": embedding}
        else:
             return {"status": "failed", "message": "No face detected"}

    except Exception as e:
        return {"status": "error", "message": str(e)}