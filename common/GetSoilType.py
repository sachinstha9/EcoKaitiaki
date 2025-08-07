from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("model/soil_type_model.h5")

def load_image(image_path, target_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

# img_path = "assets\download (4).jfif"
class_names = ['Alluvial soil', 'Black Soil', 'Cinder Soil', 'Clayey soils', 'Laterite soil', 'Loamy soil', 'Peat Soil', 'Sandy loam', 'Sandy soil', 'Yellow Soil']

def get_soil_type(img_path):
    inp_data = load_image(img_path, target_size=(224, 224))
    prediction = model.predict(inp_data)
    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class