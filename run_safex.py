#!/usr/bin/env python3
"""
SafeX - Production CLI Entry Point
Run SafeX in headless mode (no Streamlit UI) for 24/7 RTSP monitoring.

Usage:
    # Monitor mode (live cameras)
    python run_safex.py monitor --config config.yaml

    # Batch mode (process video files)
    python run_safex.py batch --input /path/to/videos --output ./output

    # Single video
    python run_safex.py process --video /path/to/video.mp4

    # With Streamlit UI
    streamlit run app.py

For systemd service, see the safex.service example at the bottom.
"""

import os
import sys
import signal
import argparse
import time
from pathlib import Path

from utils import load_config, setup_logger

logger = setup_logger("safex.main", log_file="safex.log")


def run_monitor(config_path: str):
    """
    Run continuous RTSP monitoring (headless, 24/7).
    Connects to configured cameras and runs PPE detection.
    """
    from rtsp_monitor import RTSPMonitor

    config = load_config(config_path)
    monitor = RTSPMonitor(config)

    # Graceful shutdown on SIGINT/SIGTERM
    def signal_handler(sig, frame):
        logger.info("\nShutdown signal received...")
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start monitoring
    monitor.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(10)
            # Periodic status log
            status = monitor.get_status()
            for cam_name, cam_status in status.items():
                connected = "\u2705" if cam_status["connected"] else "\u274c"
                logger.info(
                    f"  {connected} [{cam_name}] "
                    f"Frames: {cam_status['total_frames']} | "
                    f"Violations: {cam_status['total_violations']} | "
                    f"FPS: {cam_status['fps_actual']:.1f}"
                )
    except KeyboardInterrupt:
        monitor.stop()


def run_batch(input_dir: str, output_dir: str, config_path: str):
    """
    Batch process all video files in a directory.
    """
    from video_processor import batch_process

    config = load_config(config_path)
    config["output"]["output_dir"] = output_dir

    logger.info(f"Batch processing: {input_dir} -> {output_dir}")
    reports = batch_process(input_dir, config, output_dir)

    # Summary
    total_violations = sum(r.get("summary", {}).get("total_violations", 0) for r in reports)
    total_videos = len(reports)
    errors = sum(1 for r in reports if "error" in r)

    logger.info("\n" + "="*60)
    logger.info(f"  BATCH COMPLETE")
    logger.info(f"  Videos processed: {total_videos} ({errors} errors)")
    logger.info(f"  Total violations: {total_violations}")
    logger.info(f"  Output: {output_dir}")
    logger.info("="*60)

    return reports


def run_single(video_path: str, output_dir: str, config_path: str):
    """
    Process a single video file.
    """
    from detector import PPEDetector
    from video_processor import VideoProcessor

    config = load_config(config_path)
    config["output"]["output_dir"] = output_dir

    logger.info(f"Processing: {video_path}")

    detector = PPEDetector(config)
    processor = VideoProcessor(detector, config)

    def progress(current, total):
        pct = current / max(total, 1) * 100
        print(f"\r  Progress: {pct:.1f}% ({current}/{total} frames)", end="", flush=True)

    report = processor.process_video(video_path, progress_callback=progress)
    print()  # New line after progress

    # Print report
    summary = report.get("summary", {})
    logger.info("\n" + "="*60)
    logger.info(f"  RESULTS: {report['video_file']}")
    logger.info(f"  Frames analyzed: {summary.get('total_frames_processed', 0)}")
    logger.info(f"  Persons detected: {summary.get('total_persons_detected', 0)}")
    logger.info(f"  Violations found: {summary.get('total_violations', 0)}")
    logger.info(f"  Violation rate: {summary.get('violation_rate', 0):.1f}%")
    logger.info(f"  Output video: {report.get('annotated_video_path', 'N/A')}")
    logger.info(f"  Report: {report.get('report_path', 'N/A')}")
    logger.info("="*60)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="SafeX - PPE Safety Violation Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_safex.py monitor                           # Start live camera monitoring
  python run_safex.py batch --input ./videos            # Process all videos in folder
  python run_safex.py process --video footage.mp4       # Process single video
  streamlit run app.py                                  # Launch web dashboard
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Start live RTSP camera monitoring")
    monitor_parser.add_argument("--config", default="config.yaml", help="Config file path")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process video files")
    batch_parser.add_argument("--input", "-i", required=True, help="Input directory with video files")
    batch_parser.add_argument("--output", "-o", default="./output", help="Output directory")
    batch_parser.add_argument("--config", default="config.yaml", help="Config file path")

    # Single video command
    process_parser = subparsers.add_parser("process", help="Process a single video")
    process_parser.add_argument("--video", "-v", required=True, help="Video file path")
    process_parser.add_argument("--output", "-o", default="./output", help="Output directory")
    process_parser.add_argument("--config", default="config.yaml", help="Config file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Banner
    print("\n" + "="*60)
    print("  \U0001f6e1\ufe0f  SafeX - PPE Violation Detection System")
    print("  Mode:", args.command.upper())
    print("="*60 + "\n")

    if args.command == "monitor":
        run_monitor(args.config)
    elif args.command == "batch":
        run_batch(args.input, args.output, args.config)
    elif args.command == "process":
        run_single(args.video, args.output, args.config)


if __name__ == "__main__":
    main()


# ============================================================
# SYSTEMD SERVICE FILE (save as /etc/systemd/system/safex.service)
# ============================================================
"""
[Unit]
Description=SafeX PPE Violation Detection Monitor
After=network.target

[Service]
Type=simple
User=safex
WorkingDirectory=/opt/safex
ExecStart=/opt/safex/.venv/bin/python run_safex.py monitor --config config.yaml
Restart=always
RestartSec=10
Environment=SAFEX_TELEGRAM_BOT_TOKEN=your_token_here
Environment=SAFEX_TELEGRAM_CHAT_ID=your_chat_id_here

[Install]
WantedBy=multi-user.target
"""

# ============================================================
# DOCKER SUPPORT (Dockerfile)
# ============================================================
"""
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# For headless monitoring:
CMD ["python", "run_safex.py", "monitor"]

# For web dashboard:
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
