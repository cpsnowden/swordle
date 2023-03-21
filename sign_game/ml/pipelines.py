from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
# from tensorflow.keras import Model
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd

LANDMARK_NAME = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP"
]

LANDMARK_NAME_XYZ_COL = [
    name + coordinate for name in LANDMARK_NAME for coordinate in ["_X", "_Y", "_Z"]
]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """

    """
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    mean_diff = df.subtract(mean, axis='index')
    return mean_diff.divide(std, axis='index')


def create_preprocessing_pipeline() -> Pipeline:
    """

    """
    frame_normalizer = ColumnTransformer(
        [
            ("x", FunctionTransformer(normalize, feature_names_out="one-to-one"),
             make_column_selector(pattern="_(?:x|X)$")),
            ("y", FunctionTransformer(normalize, feature_names_out="one-to-one"),
             make_column_selector(pattern="_(?:y|Y)$")),
            ("z", FunctionTransformer(normalize, feature_names_out="one-to-one"),
             make_column_selector(pattern="_(?:z|Z)$")),
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    # Hack as models already trained assuming column ordering
    column_reorder = FunctionTransformer(
        lambda df: df[LANDMARK_NAME_XYZ_COL], feature_names_out=lambda a, b: LANDMARK_NAME_XYZ_COL
    )

    return Pipeline([
        ("normalize_frame", frame_normalizer),
        ("reorder_columns", column_reorder)
    ]).set_output(transform="pandas")


# def create_classifier(model: Model):
    # return KerasClassifier(model=model)
