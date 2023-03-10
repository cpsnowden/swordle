
import numpy as np
from sign_game.ml.landmarks import Landmarks

landmark = Landmarks()


class NoHandDetectedError(Exception):
    """
    Raised if no hand could be detected in a series of frames
    """
    pass


def frames_to_landmarks(frames) -> np.ndarray:

    frames_landmarks = []
    for frame in frames:
        _, landmarks_np = landmark.image_to_landmark_np(frame)
        if landmarks_np is not None:
            frames_landmarks.append(landmarks_np.flatten())
        else:
            print("WARNING - no landmarks in frame")

    if len(frames_landmarks) == 0:
        raise NoHandDetectedError

    return np.expand_dims(np.vstack(frames_landmarks), -1)


def preprocess(cv2_imgs):
    """
    Preprocessed a set of CV2 images into landmarks

    raises: NoHandDetectedError if no hand is detected in any images
    """
    print("Extracting landmarks")
    frames = frames_to_landmarks(cv2_imgs)
    print("Extracted landmarks")
    return frames
