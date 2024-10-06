import pytest
from fastapi.testclient import TestClient
from src.main import app
from pathlib import Path

client = TestClient(app)

TEST_IMAGE_PATH = "gun1.jpg"

@pytest.fixture
def sample_image():
    with open(TEST_IMAGE_PATH, "rb") as image_file:
        yield image_file

# Test para el endpoint /model_info
def test_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    assert "model_name" in response.json()
    assert "gun_detector_model" in response.json()
    assert "semantic_segmentation_model" in response.json()

# Test para el endpoint /detect_guns
def test_detect_guns(sample_image):
    files = {"file": ("sample_image.jpg", sample_image, "image/jpeg")}
    response = client.post("/detect_guns", files=files, data={"threshold": "0.5"})
    assert response.status_code == 200
    json_response = response.json()
    assert "n_detections" in json_response
    assert "boxes" in json_response
    assert "labels" in json_response

# Test para el endpoint /annotate_guns
def test_annotate_guns(sample_image):
    files = {"file": ("sample_image.jpg", sample_image, "image/jpeg")}
    response = client.post("/annotate_guns", files=files, data={"threshold": "0.5"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

# Test para el endpoint /detect_people
def test_detect_people(sample_image):
    files = {"file": ("sample_image.jpg", sample_image, "image/jpeg")}
    response = client.post("/detect_people", files=files, data={"threshold": "0.5"})
    assert response.status_code == 200
    json_response = response.json()
    assert "n_detections" in json_response
    assert "polygons" in json_response
    assert "boxes" in json_response
    assert "labels" in json_response

# Test para el endpoint /annotate_people
def test_annotate_people(sample_image):
    files = {"file": ("sample_image.jpg", sample_image, "image/jpeg")}
    response = client.post("/annotate_people", files=files, data={"threshold": "0.5", "draw_boxes": "true"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

# Test para el endpoint /detect
def test_detect(sample_image):
    files = {"file": ("sample_image.jpg", sample_image, "image/jpeg")}
    response = client.post("/detect", files=files, data={"threshold": "0.5"})
    assert response.status_code == 200
    json_response = response.json()
    assert "detection" in json_response
    assert "segmentation" in json_response

# Test para el endpoint /annotate
def test_annotate(sample_image):
    files = {"file": ("sample_image.jpg", sample_image, "image/jpeg")}
    response = client.post("/annotate", files=files, data={"threshold": "0.5"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

# Test para el endpoint /guns
def test_guns(sample_image):
    files = {"file": ("sample_image.jpg", sample_image, "image/jpeg")}
    response = client.post("/guns", files=files, data={"threshold": "0.5"})
    assert response.status_code == 200
    json_response = response.json()
    for gun in json_response:
        assert "gun_type" in gun
        assert "location" in gun
        assert "x" in gun["location"]
        assert "y" in gun["location"]

# Test para el endpoint /people
def test_people(sample_image):
    files = {"file": ("sample_image.jpg", sample_image, "image/jpeg")}
    response = client.post("/people", files=files, data={"threshold": "0.5"})
    assert response.status_code == 200
    json_response = response.json()
    for person in json_response:
        assert "person_type" in person
        assert "location" in person
        assert "x" in person["location"]
        assert "y" in person["location"]
        assert "area" in person

