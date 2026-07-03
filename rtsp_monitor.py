"""
SafeX - RTSP Live Camera Monitor
Continuous multi-camera RTSP monitoring service with:
- Auto-reconnection on stream failure
- Thread-per-camera architecture
- Real-time violation detection and alerting
- Frame buffer for smooth processing
- Health monitoring and logging
"""

import cv2
import time
import signal
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np

from utils import (
    Violation, DetectionResult, setup_logger,
    ensure_dir, format_timestamp, load_config
)
from detector import PPEDetector
from alerts import AlertManager

logger = setup_logger("safex.rtsp")


@dataclass
class CameraConfig:
    """Configuration for a single camera stream."""
    name: str
    url: str  # RTSP URL, file path, or device index
    enabled: bool = True
    fps_process: int = 2  # Frames per second to analyze
    zone: Optional[str] = None
    required_ppe: Optional[List[str]] = None


@dataclass
class CameraStatus:
    """Runtime status for a camera."""
    name: str
    connected: bool = False
    last_frame_time: float = 0
    total_frames: int = 0
    total_violations: int = 0
    last_error: str = ""
    reconnect_count: int = 0
    fps_actual: float = 0


class CameraStream:
    """
    Manages a single RTSP/camera stream with auto-reconnection.
    Runs frame capture in a separate thread for non-blocking reads.
    """

    def __init__(self, camera_config: CameraConfig, buffer_size: int = 30):
        self.config = camera_config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.status = CameraStatus(name=camera_config.name)

    def start(self):
        """Start the frame capture thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"cam-{self.config.name}",
            daemon=True
        )
        self._thread.start()
        logger.info(f"[{self.config.name}] Stream thread started")

    def stop(self):
        """Stop the frame capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._release()
        logger.info(f"[{self.config.name}] Stream stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the buffer (non-blocking)."""
        try:
            # Drain old frames, keep latest
            frame = None
            while not self.frame_buffer.empty():
                frame = self.frame_buffer.get_nowait()
            return frame
        except queue.Empty:
            return None

    def _capture_loop(self):
        """Continuously capture frames from the stream."""
        while self._running:
            if not self._connect():
                time.sleep(5)  # Wait before retry
                continue

            while self._running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"[{self.config.name}] Frame read failed, reconnecting...")
                    self.status.connected = False
                    self._release()
                    time.sleep(2)
                    break

                # Put frame in buffer (drop old if full)
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_buffer.put(frame)
                self.status.last_frame_time = time.time()
                self.status.total_frames += 1

                # Small sleep to control capture rate
                time.sleep(0.01)

    def _connect(self) -> bool:
        """Connect/reconnect to the stream."""
        try:
            source = self.config.url
            # Handle numeric device index
            try:
                source = int(source)
            except (ValueError, TypeError):
                pass

            self.cap = cv2.VideoCapture(source)

            # RTSP optimizations
            if isinstance(source, str) and "rtsp" in source.lower():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Use TCP for more reliable RTSP
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self.cap.isOpened():
                self.status.connected = True
                self.status.last_error = ""
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"[{self.config.name}] Connected: {w}x{h} @ {fps:.1f}fps")
                return True
            else:
                self.status.connected = False
                self.status.last_error = "Failed to open stream"
                self.status.reconnect_count += 1
                logger.error(f"[{self.config.name}] Connection failed (attempt {self.status.reconnect_count})")
                return False

        except Exception as e:
            self.status.connected = False
            self.status.last_error = str(e)
            self.status.reconnect_count += 1
            logger.error(f"[{self.config.name}] Connection error: {e}")
            return False

    def _release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
            self.cap = None


class RTSPMonitor:
    """
    Multi-camera RTSP monitoring service.
    Manages multiple camera streams and runs PPE detection on each.
    
    Usage:
        monitor = RTSPMonitor(config)
        monitor.start()  # Starts all cameras
        # ... runs until stop() is called
        monitor.stop()
    """

    def __init__(self, config: dict):
        self.config = config
        self.cameras: Dict[str, CameraStream] = {}
        self.detector: Optional[PPEDetector] = None
        self.alert_manager: Optional[AlertManager] = None
        self._running = False
        self._process_thread: Optional[threading.Thread] = None

        # Output
        self.output_dir = ensure_dir(
            config.get("output", {}).get("output_dir", "./output")
        )
        self.violations_dir = ensure_dir(
            config.get("output", {}).get("violation_snapshots_dir", "./output/violations")
        )

        # Violation log
        self.violation_log: List[dict] = []
        self._log_lock = threading.Lock()

        # Callback for UI updates
        self.on_frame_processed: Optional[Callable] = None
        self.on_violation_detected: Optional[Callable] = None

        self._init_cameras()

    def _init_cameras(self):
        """Initialize camera streams from config."""
        cameras_config = self.config.get("cameras", [])

        for cam_conf in cameras_config:
            if not cam_conf.get("enabled", True):
                continue

            camera = CameraConfig(
                name=cam_conf["name"],
                url=cam_conf["url"],
                enabled=cam_conf.get("enabled", True),
                fps_process=cam_conf.get("fps_process", 2),
                zone=cam_conf.get("zone"),
                required_ppe=cam_conf.get("required_ppe")
            )
            self.cameras[camera.name] = CameraStream(camera)

        logger.info(f"Initialized {len(self.cameras)} camera(s)")

    def start(self):
        """
        Start the monitoring service.
        Loads model, starts all camera threads, and begins processing.
        """
        logger.info("="*60)
        logger.info("  SafeX RTSP Monitor - Starting")
        logger.info("="*60)

        # Load detector
        logger.info("Loading PPE detection model...")
        self.detector = PPEDetector(self.config)

        # Initialize alert manager
        self.alert_manager = AlertManager(self.config)

        # Start camera streams
        for name, stream in self.cameras.items():
            stream.start()
            time.sleep(0.5)  # Stagger connections

        # Start processing loop
        self._running = True
        self._process_thread = threading.Thread(
            target=self._processing_loop,
            name="processor",
            daemon=True
        )
        self._process_thread.start()

        logger.info("Monitor is running. Press Ctrl+C to stop.")

    def stop(self):
        """Stop all cameras and processing."""
        logger.info("Stopping monitor...")
        self._running = False

        for name, stream in self.cameras.items():
            stream.stop()

        if self._process_thread:
            self._process_thread.join(timeout=10)

        # Save final report
        self._save_session_report()
        logger.info("Monitor stopped.")

    def get_status(self) -> Dict[str, dict]:
        """Get status of all cameras."""
        status = {}
        for name, stream in self.cameras.items():
            s = stream.status
            status[name] = {
                "connected": s.connected,
                "total_frames": s.total_frames,
                "total_violations": s.total_violations,
                "last_error": s.last_error,
                "reconnect_count": s.reconnect_count,
                "fps_actual": s.fps_actual
            }
        return status

    def add_camera(self, name: str, url: str, fps_process: int = 2):
        """Dynamically add a camera stream."""
        camera_config = CameraConfig(name=name, url=url, fps_process=fps_process)
        stream = CameraStream(camera_config)
        self.cameras[name] = stream
        if self._running:
            stream.start()
        logger.info(f"Added camera: {name} -> {url}")

    def remove_camera(self, name: str):
        """Remove a camera stream."""
        if name in self.cameras:
            self.cameras[name].stop()
            del self.cameras[name]
            logger.info(f"Removed camera: {name}")

    def _processing_loop(self):
        """Main processing loop - pulls frames from all cameras and runs detection."""
        frame_times: Dict[str, float] = {}

        while self._running:
            for name, stream in self.cameras.items():
                if not stream.status.connected:
                    continue

                # Rate limiting per camera
                now = time.time()
                last = frame_times.get(name, 0)
                interval = 1.0 / stream.config.fps_process
                if now - last < interval:
                    continue

                # Get latest frame
                frame = stream.get_frame()
                if frame is None:
                    continue

                frame_times[name] = now

                # Run detection
                try:
                    result = self.detector.detect_frame(
                        frame,
                        frame_number=stream.status.total_frames,
                        timestamp=now
                    )

                    # Handle violations
                    if result.violations:
                        self._handle_violations(result, frame, stream)

                    # Callback for UI
                    if self.on_frame_processed:
                        annotated = self.detector.annotate_frame(frame, result)
                        self.on_frame_processed(name, annotated, result)

                    # Update FPS
                    elapsed = now - last if last > 0 else 1
                    stream.status.fps_actual = 1.0 / elapsed

                except Exception as e:
                    logger.error(f"[{name}] Detection error: {e}")

            # Small sleep to prevent CPU spinning
            time.sleep(0.01)

    def _handle_violations(self, result: DetectionResult, frame: np.ndarray, stream: CameraStream):
        """Process detected violations: save snapshot, send alerts, log."""
        for violation in result.violations:
            stream.status.total_violations += 1
            violation.video_source = stream.config.name

            # Save snapshot
            snapshot_path = self._save_violation_snapshot(frame, violation, stream.config.name)
            violation.snapshot_path = snapshot_path

            # Send alert
            if self.alert_manager:
                self.alert_manager.send_violation_alert(
                    violation=violation,
                    camera_name=stream.config.name,
                    snapshot_path=snapshot_path
                )

            # Log violation
            with self._log_lock:
                self.violation_log.append({
                    "camera": stream.config.name,
                    "detected_at": datetime.now().isoformat(),
                    **violation.to_dict()
                })

            # Callback
            if self.on_violation_detected:
                self.on_violation_detected(stream.config.name, violation)

            logger.warning(
                f"[{stream.config.name}] VIOLATION: "
                f"{', '.join(violation.missing_ppe)} | "
                f"Severity: {violation.severity} | "
                f"Confidence: {violation.person_bbox.confidence:.0%}"
            )

    def _save_violation_snapshot(self, frame: np.ndarray, violation: Violation, camera_name: str) -> str:
        """Save annotated violation snapshot."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        missing_str = "_".join(violation.missing_ppe)
        filename = f"{camera_name}_{timestamp_str}_{missing_str}.jpg"
        filepath = self.violations_dir / filename

        # Annotate the frame area around the violation
        annotated = frame.copy()
        person = violation.person_bbox
        cv2.rectangle(
            annotated,
            (int(person.x1), int(person.y1)),
            (int(person.x2), int(person.y2)),
            (0, 0, 255), 3
        )
        label = f"VIOLATION: {', '.join(violation.missing_ppe)}"
        cv2.putText(annotated, label, (int(person.x1), int(person.y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imwrite(str(filepath), annotated)
        return str(filepath)

    def _save_session_report(self):
        """Save cumulative session report."""
        import json
        report = {
            "session_end": datetime.now().isoformat(),
            "cameras": self.get_status(),
            "total_violations": len(self.violation_log),
            "violations": self.violation_log[-100:]  # Last 100
        }
        report_path = self.output_dir / f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Session report saved: {report_path}")
