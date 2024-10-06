import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from src.predictor import GunDetector, Detection, Segmentation, annotate_detection, annotate_segmentation
from src.models import Gun, PixelLocation, Person
from src.config import get_settings

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def detect_uploadfile_guns(detector: GunDetector, file, threshold) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # Convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )
    # Crear array de numpy
    img_array = np.array(img_obj)
    detection = detector.detect_guns(img_array, threshold)
    return detection, img_array


def detect_uploadfile_people(detector: GunDetector, file, threshold) -> tuple[Segmentation, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # Convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )
    # Crear array de numpy
    img_array = np.array(img_obj)
    segmentation = detector.segment_people(img_array, threshold)
    return segmentation, img_array


def detect_uploadfile_combined(detector: GunDetector, file, threshold) -> tuple[Detection, Segmentation, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # Convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )
    # Crear array de numpy
    img_array = np.array(img_obj)
    detection = detector.detect_guns(img_array, threshold)
    segmentation = detector.segment_people(img_array, threshold)
    return detection, segmentation, img_array


@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    detection, _ = detect_uploadfile_guns(detector, file, threshold)
    return detection


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile_guns(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


# Agregando los endpoints solicitados


@app.post("/detect_people")
def detect_people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Segmentation:
    segmentation, _ = detect_uploadfile_people(detector, file, threshold)
    return segmentation


@app.post("/annotate_people")
def annotate_people(
    threshold: float = 0.5,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    segmentation, img = detect_uploadfile_people(detector, file, threshold)
    annotated_img = annotate_segmentation(img, segmentation, draw_boxes)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/detect")
def detect(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> dict:
    detection, segmentation, _ = detect_uploadfile_combined(detector, file, threshold)
    return {
        "detection": detection,
        "segmentation": segmentation,
    }


@app.post("/annotate")
def annotate(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, segmentation, img = detect_uploadfile_combined(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)
    annotated_img = annotate_segmentation(annotated_img, segmentation, draw_boxes=True)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/guns")
def guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Gun]:
    detection, _ = detect_uploadfile_guns(detector, file, threshold)
    guns = []
    for label, box in zip(detection.labels, detection.boxes):
        if label in ["pistol", "rifle"]:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            guns.append(
                Gun(gun_type=label, location=PixelLocation(x=center_x, y=center_y))
            )
    return guns


@app.post("/people")
def people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Person]:
    segmentation, _ = detect_uploadfile_people(detector, file, threshold)
    people = []
    for label, box, polygon in zip(segmentation.labels, segmentation.boxes, segmentation.polygons):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        area = int(cv2.contourArea(np.array(polygon, dtype=np.int32)))  # Convertir a entero

        people.append(
            Person(person_type=label, location=PixelLocation(x=center_x, y=center_y), area=area)
        )
    return people


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")
