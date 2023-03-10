# sign-game-server

## Deployed Environments

1. API for deployed server https://sign-game-server-yckhsn477a-uc.a.run.app/docs

### GCP

1. To see the GCP resources, ensure you are a member of https://groups.google.com/g/lewagonmelbourne/members
2. CLoud Build Console - https://console.cloud.google.com/cloud-build/dashboard?project=wagon-bootcamp-374809
3. CLoud Run Console - https://console.cloud.google.com/run?project=wagon-bootcamp-374809

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
