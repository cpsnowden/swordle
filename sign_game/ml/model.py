
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
import numpy as np
from typing import Tuple

# def initialize_model(input_shape: tuple) -> Model:

#     print("✅ model initialized")

#     return model

# def compile_model(model: Model):
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     print("✅ model compiled")
#     return model

# def train_model(model: Model,
#                 X: np.ndarray,
#                 y: np.ndarray,
#                 patience=10,
#                 validation_split=0.3) -> Tuple[Model, dict]:
#     es = EarlyStopping(patience=patience,
#                        validation_split=validation_split,
#                        restore_best_weights=True,
#                        verbose=0)

#     history = model.fit(X, y, epochs=500, callbacks=[es], verbose=0)
#     print("✅ model trained")

#     return model, history

def predict(model: Model,
            X_pred: np.ndarray):
    # model.predict()
    pass

if __name__=='__main__':
    from sign_game.ml.registry import load_model
    import cv2
    import matplotlib.pyplot as plt
    from sign_game.ml.landmarks import Landmarks
    latest_model = load_model()
    image=cv2.imread('images/C.jpg')
    landmarks = Landmarks()
    cv2_img_w_landmarks, landmark_object = landmarks.image_to_landmark(image, draw_landmarks=True)
    print(landmark_object)
    plt.imshow(cv2_img_w_landmarks)
    plt.show()
    X_pred=np.reshape(np.array(landmark_object.values), (63, 1))
