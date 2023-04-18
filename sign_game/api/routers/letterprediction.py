from tensorflow.keras import Model
from fastapi import APIRouter, UploadFile, Depends
from enum import Enum
from pydantic import BaseModel
from enum import Enum
from typing import Optional
from sign_game.api.dependencies import resolve_model
from sign_game.util.images import bytes_to_cv2, b64_frame_to_cv2
from sign_game.ml.preprocessing import preprocess, NoHandDetectedError
from sign_game.ml.model import predict
import numpy as np

router = APIRouter(prefix="/letter-prediction", tags=["Letter Prediction"])


class B64FrameRequest(BaseModel):
    b64_frame: str


class PredictionStatus(str, Enum):
    success = 'success',
    no_hand_detected = 'no_hand_detected'


class LetterPredictionResponse(BaseModel):
    predictionStatus: PredictionStatus
    prediction: Optional[str]


@router.post('/frame')
async def predict_letter_from_frame(img: UploadFile, model: Model = Depends(resolve_model)) -> LetterPredictionResponse:
    print(f"Received predict request for frame")
    contents = await img.read()
    cv2_img = bytes_to_cv2(contents)
    return process(cv2_img, model)


@router.post('/b64-frame')
def predict_letter_from_b64_frame(frame_request: B64FrameRequest, model: Model = Depends(resolve_model)) -> LetterPredictionResponse:
    print(f"Received predict request for frame")
    cv2_img = b64_frame_to_cv2(frame_request.b64_frame)
    return process(cv2_img, model)


def process(cv2_img, model) -> LetterPredictionResponse:

    img_sequence = [cv2_img]

    try:
        X_pred = preprocess(img_sequence)
    except NoHandDetectedError:
        return LetterPredictionResponse(predictionStatus=PredictionStatus.no_hand_detected)

    # Expand axis for keras
    X_pred_np = X_pred.to_numpy()[:, :, np.newaxis]

    y_pred = predict(model, X_pred_np)

    predicted_letter = y_pred[0]

    return LetterPredictionResponse(predictionStatus=PredictionStatus.success, prediction=predicted_letter)
