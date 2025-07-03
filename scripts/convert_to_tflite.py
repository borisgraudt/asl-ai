import os
import tensorflow as tf
from tensorflow import keras

# Абсолютные пути к моделям
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_PATH = os.path.join(BASE_DIR, '../models/model.keras')
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, '../models/model.tflite')
SAVEDMODEL_DIR = os.path.join(BASE_DIR, '../models/saved_model_temp')

# 1. Загрузка модели
model = keras.models.load_model(KERAS_MODEL_PATH)

# 2. Сохраняем как SavedModel (TFLite поддерживает SavedModel)
model.export(SAVEDMODEL_DIR)  # Keras 3: export для SavedModel

# 3. Конвертация в TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
tflite_model = converter.convert()

# 4. Сохраняем TFLite-модель
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f'Модель успешно конвертирована в {TFLITE_MODEL_PATH}') 