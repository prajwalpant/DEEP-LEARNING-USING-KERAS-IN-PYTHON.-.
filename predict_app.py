import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
    global model
    model = load_model("transfer_OCT.h5")
    model._make_predict_function() 
    print(" * model loaded!")
    
def preprocess_image(image, target_size):
    if image.mode !="RGB":
        image = image.convert("RGB")
   
    image = image.resize(target_size)
    image = img_to_array(image)
  
    print("***")
    image = np.expand_dims(image, axis=0)
    #image =image.reshape(150,150)"""
    print("***")
    return image

print(" * Loading Keras model....")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(150,150))
    print("***")
    print("7777777")
    #model._make_predict_function()
    prediction = model.predict_classes(processed_image)
    print("8888888888")
    print("***")
    prediction=prediction.tolist()
    print(prediction)
    
    print(prediction[0])
            
            
 
    print(':::::::')

        
        


    response = {
        'prediction': {
           'class': prediction[0],
         
        }
    }    
    return jsonify(response)
        
        
        
    
    
    







    