
import numpy as np
from sign_game.ml.landmarks import Landmarks
landmark = Landmarks()

def frames_to_landmarks(frames) -> np.ndarray:
    frames_landmarks = [landmark.image_to_landmark_np(frame)[1].flatten() for frame in frames]
    return np.expand_dims(np.vstack(frames_landmarks), -1)
