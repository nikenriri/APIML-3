from io import BytesIO
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
import requests

# Define Function
labels = ['Jamur Enoki', 'Jamur Shimeji Coklat', 'Jamur Shimeji Putih', 'Jamur Tiram']

def process(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def download_model(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)

def predict(image: Image.Image):
    loaded_model = tf.keras.models.load_model('mvp_model.h5')
    image = tf.image.resize(np.array(image), (224, 224))
    image = Image.fromarray(np.uint8(image.numpy()))
    image = image.convert("RGB")
    image = np.expand_dims(np.array(image) / 255, 0)

    hasil = loaded_model.predict(image)
    idx = hasil.argmax()
    return labels[idx]


# FASTAPI
app = FastAPI()

@app.post("/predict/image")
async def predict_fastapi(file: UploadFile = File(...)):
    image = process(await file.read())
    prediction = predict(image)
    return prediction

if __name__ == '__main__':
    # Download model from cloud storage
    download_model('https://storage.googleapis.com/asset_mushroom/mvp_model.h5', 'mvp_model.h5')

    uvicorn.run(app, host='0.0.0.0', port=8000)

