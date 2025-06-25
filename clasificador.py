import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta logs informativos de TensorFlow

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

mobilenet = MobileNetV2(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
for layer in mobilenet.layers:
    layer.trainable = False

model = Sequential([
    mobilenet,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Generadores de imagen
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'data/train',
    target_size=(128, 128) ,
    batch_size=16,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    'data/validation',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Añadir EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',     # monitorea la pérdida en validación
    patience=3,             # espera 3 épocas sin mejora
    restore_best_weights=True
)

# Entrenar el modelo
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    callbacks=[early_stop]
)

# Guardar modelo
model.save('modelo_gatos_perros.h5')
print("✅ Modelo guardado como modelo_gatos_perros.h5")

# 📈 Graficar resultados
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión por época')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida por época')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.savefig("entrenamiento_resultado.png")  # Guarda gráfico como imagen
plt.show()
