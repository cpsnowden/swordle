from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from typing import Literal


def normalize_axis(df: pd.DataFrame) -> pd.DataFrame:
    """
        Normalizes a dataframe to construct z-scores for each row
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    mean_diff = df.subtract(mean, axis='index')
    return mean_diff.divide(std, axis='index')


def create_frame_normalizer(remainder: Literal['drop', 'passthrough'] = 'drop') -> ColumnTransformer:
    """
    Creates a column transformer which will normalize handmarks per image in
    order to centralize them against the mean x,y and z positions (hand centroid)
    """
    return ColumnTransformer(
        [
            ("x", FunctionTransformer(normalize_axis, feature_names_out="one-to-one"),
             make_column_selector(pattern="_(?:x|X)$")),
            ("y", FunctionTransformer(normalize_axis, feature_names_out="one-to-one"),
             make_column_selector(pattern="_(?:y|Y)$")),
            ("z", FunctionTransformer(normalize_axis, feature_names_out="one-to-one"),
             make_column_selector(pattern="_(?:z|Z)$")),
        ],
        remainder=remainder,
        verbose_feature_names_out=False
    ).set_output(transform="pandas")


def normalize_handmarks_per_image(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize handmarks per image in order to centralize them against the
    mean x,y and z positions (hand centroid)
    """
    normalizer = create_frame_normalizer('passthrough')
    return normalizer.fit_transform(df)
