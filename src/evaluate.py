import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/dog_cat_cnn.keras")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def main():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )

    model = tf.keras.models.load_model(MODEL_PATH)

    # Collect predictions
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["cat", "dog"]))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
