from fastapi import APIRouter
from typing import List
from pydantic import BaseModel
from sign_game.util import random_letter
import base64
import numpy as np
import os
import cv2
import datetime

router = APIRouter(prefix="/letter-prediction", tags=["Letter Prediction"])

save_frames = False
class FrameSequence(BaseModel):
    frames: List[str]

@router.post('/frame-sequence')
def predict_letter_from_frame_sequence(frame_sequence: FrameSequence):

    print(f"Received predict request for {len(frame_sequence.frames)} frames")

    if save_frames:
        request_time = datetime.datetime.now()
        os.makedirs(f"data/{request_time}", exist_ok=True)
        for i, cv2_img in enumerate(b64_to_cv2(frame_sequence.frames)):
            cv2.imwrite(f"data/{request_time}/{i}.png", cv2_img)

    return { 'prediction': random_letter() }


def b64_to_cv2(frames: List[str]):
      for frame in frames:
        frame_data = frame.split(",", 2)[1]
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        yield cv2.imdecode(nparr, cv2.IMREAD_COLOR)
