from tensorflow.keras import Model, models


def load_model(model_path) -> Model:
    """
    Load a keras model from a given path
    """
    print(f"Loading model from [{model_path}]")
    model = models.load_model(model_path)
    print(f"Loaded model from [{model_path}]")
    return model
