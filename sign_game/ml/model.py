
from tensorflow.keras import Model
import numpy as np
import string
alphabet = list(string.ascii_uppercase)


def predict(model: Model, X_pred: np.ndarray) -> np.ndarray:
    print("Predicting")
    y = model.predict(X_pred, verbose=False)
    print(f"Predicted shape {y.shape}")
    predicted_classes = np.argmax(y, axis=1)
    print("Predicted classes", predicted_classes)
    predicted_letters = np.array(
        [alphabet[predicted_class] for predicted_class in predicted_classes])
    print("Predicted letters", predicted_letters)
    return predicted_letters
