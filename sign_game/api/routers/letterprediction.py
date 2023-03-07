from fastapi import APIRouter
from typing import List
from pydantic import BaseModel
from sign_game.util import random_letter

router = APIRouter(prefix="/letter-prediction", tags=["Letter Prediction"])

class FrameSequence(BaseModel):
    frames: List[str]

@router.post('/frame-sequence')
def predict_letter_from_frame_sequence(frame_sequence: FrameSequence):

    print(f"Received predict request for {len(frame_sequence.frames)} frames")

    # frame = frame_sequence.frames[10]
    # frame_data = frame.split(",")[1]
    # img_data = base64.b64decode(frame_data)

    return { 'prediction': random_letter() }
