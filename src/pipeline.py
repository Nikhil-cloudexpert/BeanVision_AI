"""
BeanVision AI inference pipeline.

Pipeline:
YOLOv8 segmentation
    -> segmentation mask
    -> background removal
    -> bounding-box crop
    -> resize to 224x224
    -> RGB conversion
    -> Hugging Face DINOv2 feature extraction
    -> CoffeeMLP classification
    -> annotated image and JSON results

Expected classes:
0: cut
1: good
2: husk
3: immature
4: parchment
5: partial-black
6: shell
"""

import os
import time
import base64
import threading
from src.quality import assess_quality
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# MLP CLASSIFIER
# Must match the architecture used during training.
# ---------------------------------------------------------

class CoffeeMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(256, 7)
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------
# LOAD YAML CONFIGURATION
# ---------------------------------------------------------

def load_config():
    config_path = ROOT / "configs" / "app.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


# ---------------------------------------------------------
# ANNOTATION COLORS
# OpenCV uses BGR color order.
# ---------------------------------------------------------

PALETTE = [
    (255, 160, 0),
    (0, 200, 120),
    (60, 120, 255),
    (200, 80, 220),
    (0, 200, 230),
    (40, 40, 255),
    (230, 200, 0)
]


# ---------------------------------------------------------
# INFERENCE PIPELINE
# ---------------------------------------------------------

class Pipeline:

    def __init__(self, cfg):

        self.cfg = cfg
        self.classes = cfg["classes"]

        if len(self.classes) != 7:
            raise ValueError(
                "The MLP expects exactly 7 classification classes."
            )

        # Avoid simultaneous inference calls.
        self.lock = threading.Lock()

        self.device = torch.device(cfg["device"])

        # Resolve model paths relative to project root.
        seg_path = ROOT / cfg["segmentation"]["model"]
        dino_path = ROOT / cfg["dinov2_dir"]
        mlp_path = ROOT / cfg["mlp_checkpoint"]

        # Validate model files.
        for path in (seg_path, mlp_path):

            if not path.is_file():
                raise FileNotFoundError(
                    f"Required model file is missing: {path}"
                )

        for filename in (
            "config.json",
            "preprocessor_config.json"
        ):

            if not (dino_path / filename).is_file():
                raise FileNotFoundError(
                    f"DINOv2 file is missing: "
                    f"{dino_path / filename}"
                )

        if not any(
            (dino_path / filename).is_file()
            for filename in (
                "model.safetensors",
                "pytorch_model.bin"
            )
        ):

            raise FileNotFoundError(
                f"DINOv2 model weights not found in {dino_path}"
            )

        print(f"Loading YOLO model: {seg_path}")

        self.yolo = YOLO(str(seg_path))

        print(f"Loading DINOv2 model: {dino_path}")

        self.proc = AutoImageProcessor.from_pretrained(
            str(dino_path),
            local_files_only=True
        )

        self.dino = AutoModel.from_pretrained(
            str(dino_path),
            local_files_only=True
        ).to(self.device).eval()

        print(f"Loading MLP classifier: {mlp_path}")

        self.mlp = CoffeeMLP()

        state_dict = torch.load(
            mlp_path,
            map_location="cpu",
            weights_only=True
        )

        self.mlp.load_state_dict(state_dict)

        self.mlp.to(self.device).eval()

        # Initial filtering configuration.
        self.min_box_side = int(
            cfg.get("min_box_side_px", 12)
        )

        self.min_mask_fill_ratio = float(
            cfg.get("min_mask_fill_ratio", 0.20)
        )

        self.min_image_area_ratio = float(
            cfg.get("min_image_area_ratio", 0.0003)
        )

        self.max_edge_aspect_ratio = float(
            cfg.get("max_edge_aspect_ratio", 2.5)
        )

        self.low_confidence_threshold = float(
            cfg.get("low_confidence_threshold", 0.6)
        )

        print("BeanVision AI pipeline loaded successfully.")
        print(f"Inference device: {self.device}")

    # -----------------------------------------------------
    # VALIDATE AND PREPARE SEGMENTATION MASK
    # -----------------------------------------------------

    @staticmethod
    def _create_mask(poly, height, width):

        if poly is None or len(poly) < 3:
            return None

        polygon = np.asarray(
            poly,
            dtype=np.float32
        )

        if not np.isfinite(polygon).all():
            return None

        polygon = np.rint(polygon).astype(np.int32)

        # Clip coordinates to image bounds.
        polygon[:, 0] = np.clip(
            polygon[:, 0],
            0,
            width - 1
        )

        polygon[:, 1] = np.clip(
            polygon[:, 1],
            0,
            height - 1
        )

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [polygon],
            1
        )

        if not np.any(mask):
            return None

        return mask

    # -----------------------------------------------------
    # CHECK WHETHER A DETECTION IS PLAUSIBLE
    # -----------------------------------------------------

    def _valid_detection(
        self,
        mask,
        box,
        image_height,
        image_width
    ):

        x1, y1, x2, y2 = box

        box_width = x2 - x1
        box_height = y2 - y1

        # Reject invalid boxes.
        if box_width <= 0 or box_height <= 0:
            return False

        # Reject extremely small detections.
        if (
            box_width < self.min_box_side
            or box_height < self.min_box_side
        ):
            return False

        box_area = box_width * box_height

        mask_crop = mask[y1:y2, x1:x2]

        mask_area = int(mask_crop.sum())

        if mask_area <= 0:
            return False

        # How much of the box is occupied by the mask?
        fill_ratio = mask_area / box_area

        if fill_ratio < self.min_mask_fill_ratio:
            return False

        # Reject extremely small masks.
        image_area = image_height * image_width

        if image_area <= 0:
            return False

        if (
            mask_area / image_area
            < self.min_image_area_ratio
        ):
            return False

        # Check whether the box touches an image boundary.
        touches_edge = (
            x1 <= 1
            or y1 <= 1
            or x2 >= image_width - 1
            or y2 >= image_height - 1
        )

        # Only reject unusually elongated edge detections.
        # This intentionally does not reject every edge detection.
        aspect_ratio = max(
            box_width / box_height,
            box_height / box_width
        )

        if (
            touches_edge
            and aspect_ratio > self.max_edge_aspect_ratio
        ):
            return False

        return True

    # -----------------------------------------------------
    # MAIN PREDICTION FUNCTION
    # -----------------------------------------------------

    def predict(self, bgr):

        start_time = time.perf_counter()

        if (
            not isinstance(bgr, np.ndarray)
            or bgr.ndim != 3
            or bgr.shape[2] != 3
            or bgr.size == 0
        ):
            raise ValueError(
                "Expected a non-empty BGR image with 3 channels."
            )

        image_height, image_width = bgr.shape[:2]

        segmentation_cfg = self.cfg["segmentation"]

        with self.lock:

            # ---------------------------------------------
            # 1. YOLO SEGMENTATION
            # ---------------------------------------------

            # A 0.5 threshold can miss a single close-up bean. Use 0.25 for
            # inference, while still allowing an explicitly lower YAML value.
            configured_conf = float(segmentation_cfg.get("conf", 0.25))
            inference_conf = min(configured_conf, 0.25)

            results = self.yolo.predict(
    bgr,
    imgsz=segmentation_cfg["imgsz"],
    conf=inference_conf,
    iou=0.35,
    max_det=100,
    verbose=False
)

            result = results[0]

            box_count = 0 if result.boxes is None else len(result.boxes)
            mask_count = 0 if result.masks is None else len(result.masks)
            print(
                f"[BeanVision DEBUG] YOLO boxes={box_count}, "
                f"masks={mask_count}, conf={inference_conf:.2f}"
            )

            annotated_image = bgr.copy()

            crops = []
            metadata = []

            if (
                result.masks is not None
                and result.boxes is not None
                and len(result.boxes) > 0
            ):

                polygons = result.masks.xy

                boxes = (
                    result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )

                detector_classes = (
                    result.boxes.cls
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                detector_confidences = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                )

                # -----------------------------------------
                # 2. VALIDATE MASKS AND EXTRACT BEAN CROPS
                # -----------------------------------------

                for index, polygon in enumerate(polygons):

                    if len(polygon) < 3:
                        print(f"[BeanVision DEBUG] Detection {index} skipped: polygon has fewer than 3 points")
                        continue

                    # Convert YOLO coordinates into image bounds.
                    coordinates = boxes[index]

                    if not np.isfinite(coordinates).all():
                        print(f"[BeanVision DEBUG] Detection {index} skipped: non-finite box coordinates")
                        continue

                    x1, y1, x2, y2 = np.rint(
                        coordinates
                    ).astype(int)

                    x1 = int(np.clip(x1, 0, image_width))
                    y1 = int(np.clip(y1, 0, image_height))
                    x2 = int(np.clip(x2, 0, image_width))
                    y2 = int(np.clip(y2, 0, image_height))

                    box = (x1, y1, x2, y2)

                    # Build a full-resolution mask.
                    mask = self._create_mask(
                        polygon,
                        image_height,
                        image_width
                    )

                    if mask is None:
                        print(f"[BeanVision DEBUG] Detection {index} skipped: invalid/empty polygon mask")
                        continue

                    # Reject implausible detections, but log why for diagnosis.
                    if not self._valid_detection(
                        mask,
                        box,
                        image_height,
                        image_width
                    ):
                        x1d, y1d, x2d, y2d = box
                        crop_mask_area = int(mask[y1d:y2d, x1d:x2d].sum()) if x2d > x1d and y2d > y1d else 0
                        print(
                            f"[BeanVision DEBUG] Detection {index} filtered: "
                            f"box={box}, mask_area_in_box={crop_mask_area}, "
                            f"detector_conf={float(detector_confidences[index]):.3f}"
                        )
                        continue

                    # -------------------------------------
                    # 3. REMOVE BACKGROUND
                    # -------------------------------------

                    # Use the segmentation mask to suppress
                    # pixels outside the detected bean.
                    bean = np.zeros_like(bgr)

                    bean[mask > 0] = bgr[mask > 0]

                    # Extract the bounding-box crop.
                    bean_crop = bean[y1:y2, x1:x2]

                    if bean_crop.size == 0:
                        continue

                    # Resize to the classifier's expected size.
                    bean_crop = cv2.resize(
                        bean_crop,
                        (224, 224),
                        interpolation=cv2.INTER_LINEAR
                    )

                    # Convert BGR to RGB.
                    rgb_crop = cv2.cvtColor(
                        bean_crop,
                        cv2.COLOR_BGR2RGB
                    )

                    crops.append(
                        Image.fromarray(rgb_crop)
                    )

                    detector_class = int(
                        detector_classes[index]
                    )

                    if isinstance(self.yolo.names, dict):
                        detector_label = self.yolo.names.get(
                            detector_class,
                            str(detector_class)
                        )
                    else:
                        detector_label = self.yolo.names[
                            detector_class
                        ]

                    metadata.append({
                        "polygon": np.asarray(
                            polygon,
                            dtype=np.int32
                        ),
                        "bbox": box,
                        "detector_label": detector_label,
                        "detector_confidence": float(
                            detector_confidences[index]
                        )
                    })

            # ---------------------------------------------
            # 4. DINOv2 FEATURE EXTRACTION
            # ---------------------------------------------

            detections = []

            print(
                f"[BeanVision DEBUG] Valid crops passed to classifier: {len(crops)}"
            )

            if crops:

                feature_batches = []

                with torch.inference_mode():

                    for start in range(0, len(crops), 32):

                        batch_images = crops[
                            start:start + 32
                        ]

                        inputs = self.proc(
                            images=batch_images,
                            return_tensors="pt"
                        ).to(self.device)

                        outputs = self.dino(**inputs)

                        # CLS token: 768-dimensional features.
                        features = (
                            outputs.last_hidden_state[:, 0, :]
                        )

                        feature_batches.append(features)

                    features = torch.cat(
                        feature_batches,
                        dim=0
                    )

                    if features.shape[1] != 768:
                        raise RuntimeError(
                            "Unexpected DINOv2 feature size: "
                            f"{features.shape[1]}"
                        )

                    # -------------------------------------
                    # 5. MLP CLASSIFICATION
                    # -------------------------------------

                    logits = self.mlp(features)

                    probabilities = torch.softmax(
                        logits,
                        dim=1
                    ).cpu().numpy()

                # -----------------------------------------
                # 6. ANNOTATE IMAGE AND BUILD RESULTS
                # -----------------------------------------

                for index, (meta, scores) in enumerate(
                    zip(metadata, probabilities)
                ):

                    predicted_index = int(
                        np.argmax(scores)
                    )

                    predicted_class = self.classes[
                        predicted_index
                    ]

                    confidence = float(
                        scores[predicted_index]
                    )

                    color = PALETTE[
                        predicted_index % len(PALETTE)
                    ]

                    polygon = meta["polygon"]

                    x1, y1, x2, y2 = meta["bbox"]

                    # Draw the segmentation outline.
                    cv2.polylines(
                        annotated_image,
                        [polygon],
                        True,
                        color,
                        2
                    )

                    label = (
                        f"{index + 1}:"
                        f"{predicted_class} "
                        f"{confidence:.2f}"
                    )

                    # Keep the text within the image.
                    text_x = max(0, min(x1, image_width - 1))
                    text_y = max(18, min(y1 - 6, image_height - 1))

                    cv2.putText(
                        annotated_image,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA
                    )

                    detections.append({
                        "id": index + 1,
                        "bbox": [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)
                        ],
                        "predicted_class": predicted_class,
                        "score": confidence,
                        "low_confidence": bool(
                            confidence
                            < self.low_confidence_threshold
                        ),
                        "detector_label": meta[
                            "detector_label"
                        ],
                        "detector_confidence": meta[
                            "detector_confidence"
                        ],
                        "scores": {
                            class_name: float(score)
                            for class_name, score in zip(
                                self.classes,
                                scores
                            )
                        }
                    })

        # ---------------------------------------------
        # 7. ENCODE ANNOTATED IMAGE
        # ---------------------------------------------

        encoded_success, buffer = cv2.imencode(
            ".jpg",
            annotated_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                88
            ]
        )

        if not encoded_success:
            raise RuntimeError(
                "Failed to encode annotated image."
            )

        # ---------------------------------------------
        # 8. CLASS COUNTS AND SUMMARY
        # ---------------------------------------------

        class_counts = {
            class_name: 0
            for class_name in self.classes
        }

        for detection in detections:
            class_counts[
                detection["predicted_class"]
            ] += 1

        # Calculate sample-level quality assessment
        quality_assessment = assess_quality(
            detections=detections,
            classes=self.classes,
            low_confidence_threshold=self.low_confidence_threshold,
        )

        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000
        )

        # ---------------------------------------------
        # 9. RETURN API RESPONSE
        # ---------------------------------------------

        return {
            "success": True,

            "image": {
                "width": image_width,
                "height": image_height
            },

            "inference": {
                "device": str(self.device),
                "duration_ms": elapsed_ms
            },

            "summary": {
                "beans_detected": len(detections)
            },

            "class_counts": class_counts,

            "detections": detections,

            "annotated_image": (
                "data:image/jpeg;base64,"
                + base64.b64encode(buffer).decode("utf-8")
            ),

            "quality_assessment": quality_assessment
        }