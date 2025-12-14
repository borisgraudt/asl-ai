import os
import shutil
import tensorflow as tf
from tensorflow import keras

# Absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_PATH = os.path.join(BASE_DIR, "../models/model.h5")
KERAS_LEGACY_MODEL_PATH = os.path.join(BASE_DIR, "../models/model.keras")
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "../models/model.tflite")

# Temporary export directory for SavedModel
SAVEDMODEL_DIR = os.path.join(BASE_DIR, "../models/saved_model_temp")

# 1) Load model (prefer .h5, fall back to legacy .keras if needed)
model_path = KERAS_MODEL_PATH if os.path.exists(KERAS_MODEL_PATH) else KERAS_LEGACY_MODEL_PATH
model = keras.models.load_model(model_path)

# 2) Export as SavedModel (required by some TFLite flows)
if os.path.exists(SAVEDMODEL_DIR):
    shutil.rmtree(SAVEDMODEL_DIR)
model.export(SAVEDMODEL_DIR)  # Keras 3: export for SavedModel

# 3) Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
tflite_model = converter.convert()

# 4) Save .tflite
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

# 5) Cleanup temp directory
shutil.rmtree(SAVEDMODEL_DIR, ignore_errors=True)

print(f"Model successfully converted to: {TFLITE_MODEL_PATH}")