from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from sklearn.metrics import classification_report

def initialize_model(input_shape: tuple) -> Model:

    print("✅ model initialized")

    return model

def compile_model(model: Model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print("✅ model compiled")
    return model

def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                patience=10,
                validation_split=0.3) -> Tuple[Model, dict]:
    es = EarlyStopping(patience=patience
                       validation_split=validation_split,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X, y, epochs=500, callbacks=[es], verbose=0)
    print("✅ model trained")

    return model, history

def predict(model: Model,
            X_pred: np.ndarray):
    pass
