from tensorflow.keras import Model
from fastapi import APIRouter, UploadFile, Depends
from typing import List
from enum import Enum
from pydantic import BaseModel
from enum import Enum
from typing import Optional
from sign_game.api.dependencies import resolve_model
from sign_game.util.images import b64_frames_to_cv2, bytes_to_cv2
from sign_game.ml.preprocessing import preprocess, NoHandDetectedError
from sign_game.ml.model import predict

router = APIRouter(prefix="/letter-prediction", tags=["Letter Prediction"])


class FrameSequence(BaseModel):
    frames: List[str]


class PredictionStatus(str, Enum):
    success = 'success',
    no_hand_detected = 'no_hand_detected'


class LetterPredictionResponse(BaseModel):
    predictionStatus: PredictionStatus
    prediction: Optional[str]


@router.post('/frame')
async def predict_letter_from_frame(img: UploadFile, model: Model = Depends(resolve_model)) -> LetterPredictionResponse:
    print(f"Received predict request for 1 frame")
    contents = await img.read()
    cv2_img = bytes_to_cv2(contents)
    return process([cv2_img], model)


@router.post('/frame-sequence')
def predict_letter_from_frame_sequence(frame_sequence: FrameSequence, model: Model = Depends(resolve_model)) -> LetterPredictionResponse:
    print(f"Received predict request for {len(frame_sequence.frames)} frames")
    cv2_imgs = b64_frames_to_cv2(frame_sequence.frames)
    return process(cv2_imgs, model)


def process(cv2_imgs, model) -> LetterPredictionResponse:
    try:
        X_pred = preprocess(cv2_imgs)
    except NoHandDetectedError:
        return LetterPredictionResponse(predictionStatus=PredictionStatus.no_hand_detected)

    y_pred = predict(model, X_pred)
    return LetterPredictionResponse(predictionStatus=PredictionStatus.success, prediction=y_pred)
