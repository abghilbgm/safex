"""
SafeX - Video Processing Module
Handles frame extraction, batch processing, and output video generation.
"""

import cv2
import os
import json
import csv
from pathlib import Path
from typing import List, Optional, Generator, Tuple
from datetime import timedelta
import numpy as np

from utils import (
    Violation, DetectionResult, setup_logger,
    ensure_dir, format_timestamp
)
from detector import PPEDetector

logger = setup_logger("safex.video")


class VideoProcessor:
    """
    Processes video files for PPE violation detection.
    Supports batch processing, real-time analysis, and report generation.
    """

    def __init__(self, detector: PPEDetector, config: dict):
        self.detector = detector
        self.config = config
        self.video_config = config.get("video", {})
        self.output_config = config.get("output", {})

        # Setup output directories
        self.output_dir = ensure_dir(self.output_config.get("output_dir", "./output"))
        self.violations_dir = ensure_dir(self.output_config.get("violation_snapshots_dir", "./output/violations"))

    def process_video(
        self,
        video_path: str,
        output_video: bool = True,
        progress_callback=None
    ) -> dict:
        """
        Process an entire video file for PPE violations.
        
        Args:
            video_path: Path to input MP4 file
            output_video: Whether to generate annotated output video
            progress_callback: Optional callback(current_frame, total_frames)
            
        Returns:
            Dictionary with processing results and statistics
        """
        video_path = Path(video_path)
        logger.info(f"Processing video: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames, {format_timestamp(duration)}")

        # Frame sampling interval
        sample_fps = self.video_config.get("frame_sample_fps", 2)
        frame_interval = max(1, int(fps / sample_fps))

        # Output video writer
        video_writer = None
        output_video_path = None
        if output_video:
            output_video_path = self.output_dir / f"{video_path.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path), fourcc, sample_fps,
                (width, height)
            )

        # Process frames
        all_violations: List[Violation] = []
        all_results: List[DetectionResult] = []
        frame_count = 0
        processed_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps

                # Run detection
                result = self.detector.detect_frame(frame, frame_count, timestamp)
                all_results.append(result)

                # Collect violations
                for violation in result.violations:
                    violation.video_source = video_path.name
                    all_violations.append(violation)

                    # Save violation snapshot
                    if self.output_config.get("save_violation_crops", True):
                        self._save_violation_snapshot(frame, violation, processed_count)

                # Write annotated frame
                if video_writer:
                    annotated = self.detector.annotate_frame(frame, result)
                    video_writer.write(annotated)

                processed_count += 1

            frame_count += 1

            # Progress callback
            if progress_callback and frame_count % 100 == 0:
                progress_callback(frame_count, total_frames)

        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()

        # Generate report
        report = self._generate_report(video_path, all_violations, all_results, {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": duration,
            "frames_processed": processed_count
        })

        if output_video_path:
            report["annotated_video_path"] = str(output_video_path)

        logger.info(f"Processing complete. Found {len(all_violations)} violations in {processed_count} frames.")

        return report

    def process_frame_generator(
        self, video_path: str
    ) -> Generator[Tuple[np.ndarray, DetectionResult], None, None]:
        """
        Generator that yields (annotated_frame, result) for streaming/real-time display.
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_fps = self.video_config.get("frame_sample_fps", 2)
        frame_interval = max(1, int(fps / sample_fps))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                result = self.detector.detect_frame(frame, frame_count, timestamp)
                annotated = self.detector.annotate_frame(frame, result)
                yield annotated, result

            frame_count += 1

        cap.release()

    def _save_violation_snapshot(self, frame: np.ndarray, violation: Violation, idx: int):
        """Save a cropped image of the violation."""
        person = violation.person_bbox
        # Add padding
        h, w = frame.shape[:2]
        pad = 20
        x1 = max(0, int(person.x1) - pad)
        y1 = max(0, int(person.y1) - pad)
        x2 = min(w, int(person.x2) + pad)
        y2 = min(h, int(person.y2) + pad)

        crop = frame[y1:y2, x1:x2]
        filename = f"violation_{idx:05d}_t{violation.timestamp:.1f}s_{'_'.join(violation.missing_ppe)}.jpg"
        cv2.imwrite(str(self.violations_dir / filename), crop)
        violation.snapshot_path = str(self.violations_dir / filename)

    def _generate_report(
        self,
        video_path: Path,
        violations: List[Violation],
        results: List[DetectionResult],
        video_info: dict
    ) -> dict:
        """Generate comprehensive violation report."""
        # Statistics
        total_persons_detected = sum(len(r.persons) for r in results)
        frames_with_violations = sum(1 for r in results if r.violations)

        # Violation breakdown
        violation_by_type = {}
        violation_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for v in violations:
            for ppe in v.missing_ppe:
                violation_by_type[ppe] = violation_by_type.get(ppe, 0) + 1
            violation_by_severity[v.severity] = violation_by_severity.get(v.severity, 0) + 1

        report = {
            "video_file": video_path.name,
            "video_info": video_info,
            "summary": {
                "total_frames_processed": video_info["frames_processed"],
                "total_persons_detected": total_persons_detected,
                "total_violations": len(violations),
                "frames_with_violations": frames_with_violations,
                "violation_rate": round(frames_with_violations / max(video_info["frames_processed"], 1) * 100, 1)
            },
            "violation_by_type": violation_by_type,
            "violation_by_severity": violation_by_severity,
            "violations": [v.to_dict() for v in violations]
        }

        # Save report
        report_format = self.output_config.get("report_format", "csv")
        report_path = self.output_dir / f"{video_path.stem}_report"

        if report_format == "json":
            with open(f"{report_path}.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
        elif report_format == "csv":
            self._save_csv_report(f"{report_path}.csv", violations)

        report["report_path"] = str(report_path) + f".{report_format}"

        return report

    def _save_csv_report(self, filepath: str, violations: List[Violation]):
        """Save violations to CSV."""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Frame", "Missing_PPE", "Severity",
                "Zone", "Confidence", "Video_Source", "Snapshot"
            ])
            for v in violations:
                writer.writerow([
                    format_timestamp(v.timestamp),
                    v.frame_number,
                    ", ".join(v.missing_ppe),
                    v.severity,
                    v.zone or "N/A",
                    f"{v.person_bbox.confidence:.2%}",
                    v.video_source,
                    v.snapshot_path or ""
                ])


def batch_process(video_dir: str, config: dict, output_dir: str = "./output") -> List[dict]:
    """
    Process all video files in a directory.
    
    Args:
        video_dir: Directory containing video files
        config: Configuration dictionary
        output_dir: Output directory for results
        
    Returns:
        List of report dictionaries
    """
    from utils import get_video_files

    config["output"]["output_dir"] = output_dir
    detector = PPEDetector(config)
    processor = VideoProcessor(detector, config)

    video_files = get_video_files(video_dir)
    logger.info(f"Found {len(video_files)} video files in {video_dir}")

    reports = []
    for i, video_file in enumerate(video_files, 1):
        logger.info(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
        try:
            report = processor.process_video(str(video_file))
            reports.append(report)
        except Exception as e:
            logger.error(f"Failed to process {video_file.name}: {e}")
            reports.append({"video_file": video_file.name, "error": str(e)})

    # Summary
    total_violations = sum(r.get("summary", {}).get("total_violations", 0) for r in reports)
    logger.info(f"\nBatch complete. {len(reports)} videos processed. Total violations: {total_violations}")

    return reports
