from io import BytesIO
import numpy as np
import tensorflow as tf
from fastapi import FastAPI,File,UploadFile
import uvicorn
from PIL import Image


#Define Function
labels = ['Jamur Enoki', 'Jamur Shimeji Coklat', 'Jamur Shimeji Putih', 'Jamur Tiram']

def process(file)-> Image.Image:
    image = image = Image.open(BytesIO(file))
    return image

def predict(image: Image.Image):
    loaded_model = tf.keras.models.load_model('https://storage.googleapis.com/asset-model/mvp_model.h5')
    image = tf.image.resize(np.array(image), (224,224))
    image = Image.fromarray(np.uint8(image.numpy()))
    image = image.convert("RGB")
    image = np.expand_dims(np.array(image)/255,0)

    hasil = loaded_model.predict(image)
    idx = hasil.argmax()
    return labels[idx]


#FASTAPI
app = FastAPI()

@app.post("/predict/image")
async def predict_fastapi(file: UploadFile = File(...)):
    image = process(await file.read())
    prediction = predict(image)
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

