
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sign_game.ml.landmarks import Landmarks
from sign_game.ml.normalization import create_frame_normalizer

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


class NoHandDetectedError(Exception):
    """
    Raised if no hand could be detected in a series of frames
    """
    pass


def create_preprocessing_pipeline() -> Pipeline:
    """
        Creates a pre-processing pipeline which normalizes a data frame
        for each row across X,Y,Z dimensions
    """
    frame_normalizer = create_frame_normalizer()
    # Backwards compatibility given models already trained on np arrays
    # assuming column ordering
    column_reorder = FunctionTransformer(
        lambda df: df[LANDMARK_NAME_XYZ_COL], feature_names_out=lambda a, b: LANDMARK_NAME_XYZ_COL
    )

    return Pipeline([
        ("normalize_frame", frame_normalizer),
        ("reorder_columns", column_reorder)
    ]).set_output(transform="pandas")


single_image_landmarks = Landmarks(static_image_mode=True)


def frames_to_landmarks(frames) -> pd.DataFrame:
    """
        Converts a set of frames to a data frame of landmarks
    """

    landmarks = []
    for frame in frames:
        _, frame_landmarks = single_image_landmarks.image_to_landmark(frame)
        if frame_landmarks is not None:
            landmarks.append(frame_landmarks)
        else:
            print("WARNING - no landmarks in frame")

    if len(landmarks) == 0:
        raise NoHandDetectedError

    return pd.DataFrame(landmarks)


def preprocess(cv2_imgs) -> pd.DataFrame:
    """
    Preprocessed a set of CV2 images into landmarks

    raises: NoHandDetectedError if no hand is detected in any images
    """
    print("Extracting landmarks")
    landmarks = frames_to_landmarks(cv2_imgs)
    print("✅ Extracted landmarks with shape", landmarks.shape)

    print("Preprocessing")
    preprocessor = create_preprocessing_pipeline()
    X_processed = preprocessor.fit_transform(landmarks)
    print("✅ Preprocessed with shape", X_processed.shape)
    return X_processed
