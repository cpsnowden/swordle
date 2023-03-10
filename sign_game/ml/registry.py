from tensorflow.keras import Model, models
import os
from enum import Enum
import mlflow
import time
from sign_game.util.params import MODELS_LOCAL_REPOSITORY, MODELS_MLFLOW_TRACKING_URI

def load_model(model_name='baseline_model') -> Model:
    model_directory = os.path.join('/models/', model_name)
    latest_model = models.load_model(model_directory)
    return latest_model


class ModelSaveLocation(Enum):
    LOCAL = 1
    MLFLOW = 2


def save_model(model: Model, model_name: str, location: ModelSaveLocation):
    """
    Save a model to a given location either Local or MFLow
    """
    if location == ModelSaveLocation.MLFLOW:
        print("Saving model to MLFlow")
        mlflow.set_tracking_uri(MODELS_MLFLOW_TRACKING_URI)
        mlflow.tensorflow.save_model(model=model,
                                    artifact_path="model",
                                    registered_model_name=model_name)
        print("Saved model to MLFlow")

    elif location == ModelSaveLocation.LOCAL:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(MODELS_LOCAL_REPOSITORY, model_name,
                                  timestamp)

        print(f"Saving model to path {model_path}")
        model.save(model_path)
        print(f"Saved model to path {model_path}")

    raise ValueError(f"Unknown save location {location}")
