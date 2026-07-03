"""
SafeX - Utility Functions
Common utilities for the PPE violation detection system.
"""

import os
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np


# ============================================================
# Logging Setup
# ============================================================
def setup_logger(name: str = "safex", log_file: Optional[str] = None) -> logging.Logger:
    """Configure structured logging."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# Data Classes
# ============================================================
@dataclass
class BoundingBox:
    """Represents a bounding box detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "confidence": round(self.confidence, 3),
            "class_id": self.class_id,
            "class_name": self.class_name
        }


@dataclass
class Violation:
    """Represents a PPE violation event."""
    timestamp: float  # seconds in video
    frame_number: int
    person_bbox: BoundingBox
    missing_ppe: List[str]
    severity: str  # critical, high, medium
    zone: Optional[str] = None
    snapshot_path: Optional[str] = None
    video_source: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp_sec": round(self.timestamp, 2),
            "timestamp_fmt": str(timedelta(seconds=int(self.timestamp))),
            "frame_number": self.frame_number,
            "person_bbox": self.person_bbox.to_dict(),
            "missing_ppe": self.missing_ppe,
            "severity": self.severity,
            "zone": self.zone,
            "snapshot_path": self.snapshot_path,
            "video_source": self.video_source
        }


@dataclass
class DetectionResult:
    """Aggregated detection results for a single frame."""
    frame_number: int
    timestamp: float
    persons: List[BoundingBox] = field(default_factory=list)
    helmets: List[BoundingBox] = field(default_factory=list)
    no_helmets: List[BoundingBox] = field(default_factory=list)
    vests: List[BoundingBox] = field(default_factory=list)
    no_vests: List[BoundingBox] = field(default_factory=list)
    safety_shoes: List[BoundingBox] = field(default_factory=list)
    no_safety_shoes: List[BoundingBox] = field(default_factory=list)
    violations: List[Violation] = field(default_factory=list)


# ============================================================
# Configuration
# ============================================================
def load_config(config_path: str = "config.yaml", cameras_config_path: str = "config_cameras.yaml") -> dict:
    """Load YAML configuration files. Merges camera config if present."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Merge camera & alert config if available
    cameras_path = Path(cameras_config_path)
    if cameras_path.exists():
        with open(cameras_path, "r") as f:
            cameras_config = yaml.safe_load(f)
        if cameras_config:
            config.update(cameras_config)

    return config


# ============================================================
# Geometry Utilities
# ============================================================
def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Intersection over Union between two boxes."""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1.area
    area2 = box2.area
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def is_inside(inner_box: BoundingBox, outer_box: BoundingBox, threshold: float = 0.6) -> bool:
    """Check if inner_box is mostly inside outer_box."""
    x1 = max(inner_box.x1, outer_box.x1)
    y1 = max(inner_box.y1, outer_box.y1)
    x2 = min(inner_box.x2, outer_box.x2)
    y2 = min(inner_box.y2, outer_box.y2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = inner_box.area

    return (intersection / inner_area) >= threshold if inner_area > 0 else False


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """Check if a point is inside a polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def get_severity(missing_ppe: List[str], severity_config: dict) -> str:
    """Determine violation severity based on missing PPE items."""
    for severity in ["critical", "high", "medium"]:
        if any(item in severity_config.get(severity, []) for item in missing_ppe):
            return severity
    return "low"


# ============================================================
# File Utilities
# ============================================================
def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_video_files(directory: str, extensions: List[str] = None) -> List[Path]:
    """Get all video files in a directory."""
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

    directory = Path(directory)
    video_files = []
    for ext in extensions:
        video_files.extend(directory.glob(f"*{ext}"))
        video_files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(video_files)


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))
