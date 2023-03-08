import base64
from typing import List

import cv2
import numpy as np


def b64_frames_to_cv2(b64_frames: List[str]):
    """
        Convert a list of b64 images to CV2 color images

        Returns: a generator of images
    """
    for frame in b64_frames:
        yield b64_frame_to_cv2(frame)


def b64_frame_to_cv2(b64_frame: str):
    """
        Convert a single b64 image to CV2 color image
    """
    frame_data = b64_frame.split(",", 2)[1]
    frame_bytes = base64.b64decode(frame_data)
    return bytes_to_cv2(frame_bytes)

def bytes_to_cv2(bytes: bytes):
    nparr = np.frombuffer(bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
