
# The Creation of Swordle: Building a CNN Classifier with Mediapipe and Keras to identify American Sign Language Fingerspelling

## Introduction

As part of our final project for the 9 week Data Science bootcamp with Le Wagon, our team created a CNN classifier to identify different American Sign Language Fingerspelling letters packaged into a word game to allow the user to practice their signing in American Sign Languge. We chose American Sign Language as this was the language that we were able to find the most data for, but given more resources wand a longer time frame we would have liked to source more accurate Auslan data and expanded out approach to include words as well as individual letters.

![Flowchart](flowchart.png)

The above flowchart shows the process that information follows to create the prediction:

1. Image capture from frontend camera
2. Frontend sends to web server via FastAPI
3. Web server applies Mediapipe coordinate transformation to image
4. Data is normalized and preprocessed
5. Pre-processed data is passed to model for prediction
6. Model predicts based on input data
7. Prediction sent from web-server back to frontend via FastAPI

Our product is packaged across two repos, sign-game-server and sign-game-UI.


---
# Instructions to Run

## Developing Locally

#### Create VENV

```bash
pyenv virtualenv sign-game-server
pyenv activate sign-game-server
pyenv local sign-game-server
```

#### Install Requirements

```bash
pip install -r requirements.txt
```

#### Start Local Fast API

```bash
make run
```

#### Running Local API with New Keras Model

1. Update .env with the path to the new model
2. Run `make run`
3. Check that you model is picked up at http://localhost:8000/info

## Converting images to landmark data - using scripts - Landmarks.py

1. Structure folder as per the following:

```bash
---FOLDER---
Landmarks.py
---asl_dataset---
|
- a
|
- b
|
- c
...
```

2. Uncomment the relevant lines in Landmarks.py

```python
if __name__ == '__main__':
    image_path = './asl_dataset/a/hand1_a_bot_seg_1_cropped.jpeg'
    video_path = './asl_dataset_videos/J/1.avi'
    landmarks = Landmarks()
    # landmarks.get_image_with_landmarks(image_path)
    # landmarks.video_to_landmark(video_path)
    # landmarks.create_csv_from_dataset_folder()
```

get_image_with_landmarks() - converts a single image from jpeg to landmark

video_to_landmark() - converts a single video to timeseries data without a target column, with rows representing separate time points

create_csv_from_landmarks() - converts a folder structured as above into a csv dataset with relevant landmark data alongside target column and path to original image

3. Run the .py script

```bash
python Landmarks.py
```

4. CSV file will be generated in the same folder

## Converting dataset to landmark data - using import into a notebook

1. Ensure folder is structured as above

2. Run:

```python
from sign_game.ml.landmarks import Landmarks()
```

3. Instantiate landmark object and run a method as above

```python
landmarks = Landmarks()
```

## Splitting a CSV - using data-split.py

1. put the csv to be split into the same folder as data-split.py

2. update the path in data-split.py as below to include the name of the csv to be split

```python
if __name__ == '__main__':
    csv_path = 'images_ds.csv'
    csv_train_test_split(csv_path, 0.2)
```

3. run

```bash
python data-split.py
```

4. Enjoy your split CSV :)

## Deployed Environments

1. API for deployed server https://sign-game-server-yckhsn477a-uc.a.run.app/docs

### GCP

1. To see the GCP resources, ensure you are a member of https://groups.google.com/g/lewagonmelbourne/members
2. Cloud Build Console - https://console.cloud.google.com/cloud-build/dashboard?project=wagon-bootcamp-374809
3. Cloud Run Console - https://console.cloud.google.com/run?project=wagon-bootcamp-374809
