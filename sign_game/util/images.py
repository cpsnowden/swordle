import base64

import cv2
import numpy as np


def b64_frame_to_cv2(b64_frame: str):
    """
        Convert a single b64 image to CV2 color image
    """
    frame_data = b64_frame.split(",", 2)[1]
    frame_bytes = base64.b64decode(frame_data)
    return bytes_to_cv2(frame_bytes)


def bytes_to_cv2(frame_bytes: bytes):
    """
        Convert a frame byte to a CV2 color image
    """
    nparr = np.frombuffer(frame_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
