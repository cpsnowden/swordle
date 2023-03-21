# TODO
# run -> starts the local uvicorn web-server - gives access to prediction API
run:
	uvicorn sign_game.api.fast:app --host 0.0.0.0 --reload

docker_build:
	docker build --tag sign-game-server:dev .
