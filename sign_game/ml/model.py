
from tensorflow.keras import Model
import numpy as np
import string
alphabet = list(string.ascii_uppercase)


def predict(model: Model, X_pred: np.ndarray):
    print("Predicting")
    y = model.predict(X_pred, verbose=False)
    print(f"Predicted shape {y.shape}")
    predicted_classes = np.argmax(y, axis=1)
    print("Predicted classes", predicted_classes)
    predicted_class = np.bincount(predicted_classes).argmax()
    predicted_letter = alphabet[predicted_class]
    print("Predicted class", predicted_class, predicted_letter)

    return predicted_letter
