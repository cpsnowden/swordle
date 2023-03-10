from tensorflow.keras import Model
from fastapi import APIRouter, UploadFile, Depends, HTTPException
from typing import List
from pydantic import BaseModel
from enum import Enum
from sign_game.api.dependencies import resolve_model
from sign_game.util.images import b64_frames_to_cv2, bytes_to_cv2
from sign_game.ml.preprocessing import preprocess, NoHandDetectedError
from sign_game.ml.model import predict

router = APIRouter(prefix="/letter-prediction", tags=["Letter Prediction"])


class FrameSequence(BaseModel):
    frames: List[str]


class LetterPredictionResponse(BaseModel):
    prediction: str


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
        raise HTTPException(400, "No hand detected!")

    y_pred = predict(model, X_pred)
    return LetterPredictionResponse(prediction=y_pred)


# save_frames = False

# def process(cv2_imgs) -> LetterPredictionResponse:

#     request_time = datetime.datetime.now()
#     for i, cv2_img in enumerate(cv2_imgs):
#         # Move this inside a pipeline - it is model preprocessing specific...
#         cv2_img_w_landmarks, landmark_dict = landmarks.image_to_landmark(
#             cv2_img, draw_landmarks=save_frames)
#         pprint(landmark_dict)
#         if landmark_dict is None:
#             print(f"No landmark found in frame {i}")
#         if save_frames:
#             os.makedirs(f"data/{request_time}", exist_ok=True)
#             cv2.imwrite(f"data/{request_time}/{i}.png", cv2_img)
#             cv2.imwrite(
#                 f"data/{request_time}/{i}_landmarks.png", cv2_img_w_landmarks)

#     return LetterPredictionResponse(prediction=random_letter())
