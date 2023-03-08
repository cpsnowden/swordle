FROM python:3.10.6-buster
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y python3-opencv

COPY sign_game sign_game
CMD uvicorn sign_game.api.fast:app --host 0.0.0.0 --port $PORT
