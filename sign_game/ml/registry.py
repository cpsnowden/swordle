from tensorflow.keras import Model, models
import os

# TODO Support loading from MLFlow
def load_model(model_path)-> Model:
    print(f"Loading model from [{model_path}]")
    model = models.load_model(model_path)
    print(f"Loaded model from [{model_path}]")
    return model
