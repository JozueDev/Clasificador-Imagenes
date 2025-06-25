from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


model = load_model('modelo_gatos_perros.h5')


def cargar_y_procesar(ruta):
    img = image.load_img(ruta, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0


ruta_imagen = "data/validation/1.jpg"
img_proc = cargar_y_procesar(ruta_imagen)

# Predecir
pred = model.predict(img_proc)[0][0]
print(f"🔎 Resultado: {'Perro' if pred >= 0.5 else 'Gato'} ({pred:.2f})")
