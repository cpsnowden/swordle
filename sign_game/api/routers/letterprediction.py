from fastapi import APIRouter
from typing import List
from pydantic import BaseModel
from sign_game.util.images import b64_frames_to_cv2
from sign_game.ml.landmarks import Landmarks
from sign_game.util import random_letter
from pprint import pprint
import os
import cv2
import datetime

router = APIRouter(prefix="/letter-prediction", tags=["Letter Prediction"])

save_frames = False
class FrameSequence(BaseModel):
    frames: List[str]

landmarks = Landmarks()

@router.post('/frame-sequence')
def predict_letter_from_frame_sequence(frame_sequence: FrameSequence):

    print(f"Received predict request for {len(frame_sequence.frames)} frames")

    request_time = datetime.datetime.now()
    for i, cv2_img in enumerate(b64_frames_to_cv2(frame_sequence.frames)):
        # Move this inside a pipeline - it is model preprocessing specific...
        cv2_img_w_landmarks, landmark_dict = landmarks.image_to_landmark(cv2_img, draw_landmarks=save_frames)
        pprint(landmark_dict)
        if landmark_dict is None:
            print(f"No landmark found in frame {i}")
        if save_frames:
            os.makedirs(f"data/{request_time}", exist_ok=True)
            cv2.imwrite(f"data/{request_time}/{i}.png", cv2_img)
            cv2.imwrite(f"data/{request_time}/{i}_landmarks.png", cv2_img_w_landmarks)

    return { 'prediction': random_letter() }
