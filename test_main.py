import httpx
import pytest

# The base URL of the running FastAPI application
BASE_URL = "http://localhost:8001"

@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL) as client:
        yield client

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_get_recommendations(client):
    response = client.post(
        "/recommendations/",
        json={"song_name": "hey, soul sister", "artist_name": "train", "recommender_type": "collaborative"},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)
