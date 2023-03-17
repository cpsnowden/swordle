
import numpy as np
import pandas as pd
from sign_game.ml.landmarks import Landmarks
from sign_game.ml.landmarks_utils import normalize_handmarks_per_image


class NoHandDetectedError(Exception):
    """
    Raised if no hand could be detected in a series of frames
    """
    pass


def frames_to_landmarks(frames) -> np.ndarray:

    frames_landmarks = []
    landmark = Landmarks()
    for frame in frames:
        _, landmarks = landmark.image_to_landmark(frame)
        if landmarks is not None:
            norm_landmark = normalize_handmarks_per_image(
                pd.DataFrame.from_dict([landmarks]))
            landmarks = norm_landmark.to_numpy()
            frames_landmarks.append(landmarks.flatten())
        else:
            print("WARNING - no landmarks in frame")

    if len(frames_landmarks) == 0:
        raise NoHandDetectedError

    res = np.expand_dims(np.vstack(frames_landmarks), -1)

    return res


def preprocess(cv2_imgs):
    """
    Preprocessed a set of CV2 images into landmarks

    raises: NoHandDetectedError if no hand is detected in any images
    """
    print("Extracting landmarks")
    frames = frames_to_landmarks(cv2_imgs)
    print("Extracted landmarks")
    return frames
