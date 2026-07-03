"""
SafeX - PPE Violation Detection Engine
Core detection logic using YOLOv8 for PPE compliance monitoring.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO

from utils import (
    BoundingBox, Violation, DetectionResult,
    is_inside, get_severity, point_in_polygon,
    setup_logger, ensure_dir
)

logger = setup_logger("safex.detector")


class PPEDetector:
    """
    YOLOv8-based PPE (Personal Protective Equipment) violation detector.
    
    Supports two modes:
    1. Fine-tuned PPE model: Directly detects helmet/vest/shoes classes
    2. Base YOLO + heuristics: Detects persons, uses region-based heuristics
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_config = config["model"]
        self.ppe_classes = config.get("ppe_classes", {})
        self.violation_rules = config.get("violation_rules", {})
        self.zones_config = config.get("zones", {"enabled": False})

        # Load model
        self.model = self._load_model()
        self.class_names = self.model.names
        self.is_ppe_model = self._check_if_ppe_model()

        logger.info(f"Model loaded: {self.model_config['weights']}")
        logger.info(f"PPE-specific model: {self.is_ppe_model}")
        logger.info(f"Classes: {self.class_names}")

    def _load_model(self) -> YOLO:
        """Load YOLOv8 model from weights."""
        weights = self.model_config["weights"]
        device = self.model_config.get("device", "auto")

        model = YOLO(weights)

        if device != "auto":
            model.to(device)

        return model

    def _check_if_ppe_model(self) -> bool:
        """Check if the loaded model has PPE-specific classes."""
        ppe_keywords = ["helmet", "vest", "shoe", "hardhat", "goggles", "glove"]
        model_classes = [name.lower() for name in self.class_names.values()]
        return any(keyword in cls for cls in model_classes for keyword in ppe_keywords)

    def detect_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> DetectionResult:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR image as numpy array
            frame_number: Frame index in video
            timestamp: Timestamp in seconds
            
        Returns:
            DetectionResult with all detections and violations
        """
        result = DetectionResult(frame_number=frame_number, timestamp=timestamp)

        # Run YOLO inference
        predictions = self.model(
            frame,
            conf=self.model_config["confidence_threshold"],
            iou=self.model_config["iou_threshold"],
            imgsz=self.model_config.get("img_size", 640),
            verbose=False
        )

        # Parse detections
        detections = self._parse_detections(predictions)

        if self.is_ppe_model:
            # PPE model: directly classify PPE presence/absence
            result = self._process_ppe_detections(detections, result)
        else:
            # Base YOLO: detect persons, use spatial heuristics
            result = self._process_base_detections(detections, result)

        # Check violations
        result.violations = self._check_violations(result)

        return result

    def _parse_detections(self, predictions) -> List[BoundingBox]:
        """Parse YOLO predictions into BoundingBox objects."""
        detections = []

        for pred in predictions:
            boxes = pred.boxes
            for box in boxes:
                bbox = BoundingBox(
                    x1=float(box.xyxy[0][0]),
                    y1=float(box.xyxy[0][1]),
                    x2=float(box.xyxy[0][2]),
                    y2=float(box.xyxy[0][3]),
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                    class_name=self.class_names[int(box.cls[0])]
                )
                detections.append(bbox)

        return detections

    def _process_ppe_detections(self, detections: List[BoundingBox], result: DetectionResult) -> DetectionResult:
        """Process detections from a PPE-specific model."""
        ppe_mapping = self.ppe_classes

        for det in detections:
            name = det.class_name.lower()

            # Categorize each detection
            if name == "person" or det.class_id == 0:
                result.persons.append(det)
            elif any(name == v.get("present_class", "").lower() for v in ppe_mapping.values() if "helmet" in v.get("present_class", "").lower() or v.get("present_class", "").lower() == name):
                # Check which PPE category this belongs to
                for ppe_type, classes in ppe_mapping.items():
                    if name == classes.get("present_class", "").lower():
                        if ppe_type == "helmet":
                            result.helmets.append(det)
                        elif ppe_type == "vest":
                            result.vests.append(det)
                        elif ppe_type == "safety_shoes":
                            result.safety_shoes.append(det)
                    elif name == classes.get("absent_class", "").lower():
                        if ppe_type == "helmet":
                            result.no_helmets.append(det)
                        elif ppe_type == "vest":
                            result.no_vests.append(det)
                        elif ppe_type == "safety_shoes":
                            result.no_safety_shoes.append(det)

        return result

    def _process_base_detections(self, detections: List[BoundingBox], result: DetectionResult) -> DetectionResult:
        """Process detections from base YOLO model (person detection only)."""
        person_config = self.config.get("person_detection", {})
        min_area = person_config.get("min_area", 5000)

        for det in detections:
            if det.class_id == person_config.get("class_id", 0):
                if det.area >= min_area:
                    result.persons.append(det)

        return result

    def _check_violations(self, result: DetectionResult) -> List[Violation]:
        """Determine PPE violations based on detections."""
        violations = []
        required_ppe = self.violation_rules.get("required_ppe", [])
        severity_config = self.violation_rules.get("severity_levels", {})

        for person in result.persons:
            missing_ppe = []

            if self.is_ppe_model:
                # Check if person has required PPE items nearby
                if "helmet" in required_ppe:
                    has_helmet = any(
                        is_inside(h, person, threshold=0.3) or self._is_above(h, person)
                        for h in result.helmets
                    )
                    has_no_helmet = any(
                        is_inside(h, person, threshold=0.3) or self._is_above(h, person)
                        for h in result.no_helmets
                    )
                    if has_no_helmet or not has_helmet:
                        missing_ppe.append("helmet")

                if "vest" in required_ppe:
                    has_vest = any(
                        is_inside(v, person, threshold=0.3)
                        for v in result.vests
                    )
                    has_no_vest = any(
                        is_inside(v, person, threshold=0.3)
                        for v in result.no_vests
                    )
                    if has_no_vest or not has_vest:
                        missing_ppe.append("vest")

                if "safety_shoes" in required_ppe:
                    has_shoes = any(
                        is_inside(s, person, threshold=0.2) or self._is_below(s, person)
                        for s in result.safety_shoes
                    )
                    has_no_shoes = any(
                        is_inside(s, person, threshold=0.2) or self._is_below(s, person)
                        for s in result.no_safety_shoes
                    )
                    if has_no_shoes or not has_shoes:
                        missing_ppe.append("safety_shoes")
            else:
                # Base model: flag all persons as potential violations
                # (since we can't detect PPE without a fine-tuned model)
                missing_ppe = ["helmet", "vest", "safety_shoes"]

            if missing_ppe:
                # Check zone constraints
                zone = self._get_zone(person) if self.zones_config.get("enabled") else None

                violation = Violation(
                    timestamp=result.timestamp,
                    frame_number=result.frame_number,
                    person_bbox=person,
                    missing_ppe=missing_ppe,
                    severity=get_severity(missing_ppe, severity_config),
                    zone=zone
                )
                violations.append(violation)

        return violations

    def _is_above(self, box: BoundingBox, person: BoundingBox) -> bool:
        """Check if box is in the upper region of person (helmet area)."""
        person_top_region = person.y1 + (person.y2 - person.y1) * 0.25
        return box.center[1] < person_top_region and box.center[0] > person.x1 and box.center[0] < person.x2

    def _is_below(self, box: BoundingBox, person: BoundingBox) -> bool:
        """Check if box is in the lower region of person (shoes area)."""
        person_bottom_region = person.y1 + (person.y2 - person.y1) * 0.75
        return box.center[1] > person_bottom_region and box.center[0] > person.x1 and box.center[0] < person.x2

    def _get_zone(self, person: BoundingBox) -> Optional[str]:
        """Determine which zone a person is in."""
        if not self.zones_config.get("enabled"):
            return None

        center = person.center
        for zone in self.zones_config.get("areas", []):
            if point_in_polygon(center, zone["polygon"]):
                return zone["name"]

        return None

    def annotate_frame(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Draw detection boxes and violation annotations on frame.
        
        Args:
            frame: Original BGR frame
            result: Detection results for this frame
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw person boxes
        for person in result.persons:
            has_violation = any(
                v.person_bbox == person for v in result.violations
            )
            color = (0, 0, 255) if has_violation else (0, 255, 0)
            thickness = 3 if has_violation else 2

            cv2.rectangle(
                annotated,
                (int(person.x1), int(person.y1)),
                (int(person.x2), int(person.y2)),
                color, thickness
            )

        # Draw PPE detections
        for helmet in result.helmets:
            cv2.rectangle(annotated, (int(helmet.x1), int(helmet.y1)),
                         (int(helmet.x2), int(helmet.y2)), (0, 255, 0), 2)
            cv2.putText(annotated, f"Helmet {helmet.confidence:.0%}",
                       (int(helmet.x1), int(helmet.y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for no_helmet in result.no_helmets:
            cv2.rectangle(annotated, (int(no_helmet.x1), int(no_helmet.y1)),
                         (int(no_helmet.x2), int(no_helmet.y2)), (0, 0, 255), 2)
            cv2.putText(annotated, f"NO HELMET {no_helmet.confidence:.0%}",
                       (int(no_helmet.x1), int(no_helmet.y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for vest in result.vests:
            cv2.rectangle(annotated, (int(vest.x1), int(vest.y1)),
                         (int(vest.x2), int(vest.y2)), (0, 255, 0), 2)
            cv2.putText(annotated, f"Vest {vest.confidence:.0%}",
                       (int(vest.x1), int(vest.y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for no_vest in result.no_vests:
            cv2.rectangle(annotated, (int(no_vest.x1), int(no_vest.y1)),
                         (int(no_vest.x2), int(no_vest.y2)), (0, 0, 255), 2)
            cv2.putText(annotated, f"NO VEST {no_vest.confidence:.0%}",
                       (int(no_vest.x1), int(no_vest.y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw violation labels
        for violation in result.violations:
            person = violation.person_bbox
            label = f"VIOLATION: {', '.join(violation.missing_ppe)}"
            severity_color = {
                "critical": (0, 0, 255),
                "high": (0, 128, 255),
                "medium": (0, 255, 255)
            }.get(violation.severity, (255, 255, 255))

            # Background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated,
                         (int(person.x1), int(person.y1) - text_size[1] - 10),
                         (int(person.x1) + text_size[0], int(person.y1)),
                         severity_color, -1)
            cv2.putText(annotated, label,
                       (int(person.x1), int(person.y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw timestamp
        from utils import format_timestamp
        ts_label = f"Time: {format_timestamp(result.timestamp)} | Frame: {result.frame_number}"
        cv2.putText(annotated, ts_label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw violation count
        violation_label = f"Violations: {len(result.violations)}"
        color = (0, 0, 255) if result.violations else (0, 255, 0)
        cv2.putText(annotated, violation_label, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return annotated
