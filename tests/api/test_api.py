
from fastapi.testclient import TestClient
from sign_game.api.fast import app

client = TestClient(app)


def test_root_api():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong"}


def test_info_api():
    response = client.get("/info")
    assert response.status_code == 200

    info = response.json()
    assert "current_model" in info
    assert info["current_model"] is not None
