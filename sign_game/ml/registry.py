from tensorflow.keras import Model, models
import os

def load_model(model_name='baseline_model')-> Model:
    model_directory=os.path.join('/models/',model_name)
    latest_model=models.load_model(model_directory)
    return latest_model
