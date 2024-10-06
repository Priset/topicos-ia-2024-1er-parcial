from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation, PersonType
from src.config import get_settings

SETTINGS = get_settings()


def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    segment_box = box(*segment)
    closest_box = None
    min_distance = float('inf')

    for bbox in bboxes:
        gun_box = box(*bbox)
        distance = segment_box.distance(gun_box)
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            closest_box = bbox

    return closest_box if min_distance <= max_distance else None


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2,y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img


def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    annotated_img = image_array.copy()

    # Opacidad para la transparencia (entre 0 y 1)
    alpha = 0.5

    for label, box, polygon in zip(segmentation.labels, segmentation.boxes, segmentation.polygons):
        # Asignar colores de acuerdo a la etiqueta
        if label == PersonType.danger:
            color = (255, 0, 0)  # Rojo para 'danger'
        else:
            color = (0, 255, 0)  # Verde para 'safe'

        # Dibujar el polígono (relleno con transparencia)
        polygon_points = np.array(polygon, dtype=np.int32)
        overlay = annotated_img.copy()
        cv2.fillPoly(overlay, [polygon_points], color)

        # Aplicar la transparencia mezclando la imagen original con la superposición
        cv2.addWeighted(overlay, alpha, annotated_img, 1 - alpha, 0, annotated_img)

        # Dibujar el bounding box si se requiere
        if draw_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_img,
                label,  # Utilizamos directamente el valor del string
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

    return annotated_img


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10):
        results = self.seg_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        person_indexes = [i for i in range(len(labels)) if labels[i] == 0]  # 0 = "person"

        polygons = [
            [[int(x), int(y)] for x, y in mask]
            for i, mask in enumerate(results.masks.xy) if i in person_indexes
        ]
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in person_indexes
        ]

        # Obtener las armas detectadas para etiquetar a las personas
        gun_detection = self.detect_guns(image_array, threshold)
        gun_boxes = gun_detection.boxes

        labels_txt = []
        for box in boxes:
            closest_gun = match_gun_bbox(box, gun_boxes, max_distance)
            if closest_gun:
                labels_txt.append(PersonType.danger)
            else:
                labels_txt.append(PersonType.safe)

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(boxes),
            polygons=polygons,
            boxes=boxes,
            labels=labels_txt,
        )
