from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_info():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello world from Codebook Automation API!"
