import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from pathlib import Path
import sys

MODEL_PATH = Path("models/dog_cat_cnn.keras")
IMG_SIZE = (128, 128)

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        return

    image_path = sys.argv[1]
    model = tf.keras.models.load_model(MODEL_PATH)

    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0  # match training normalization

    pred = model.predict(arr)[0][0]
    label = "dog" if pred > 0.5 else "cat"
    conf = pred if pred > 0.5 else 1 - pred

    print(f"Prediction: {label} (confidence: {conf:.2%})")

if __name__ == "__main__":
    main()
