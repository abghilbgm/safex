"""
SafeX - PPE Safety Violation Detection App
Industry-grade Streamlit application for real-time and batch
PPE compliance monitoring from plant camera footage.

Run: streamlit run app.py
"""

import streamlit as st
import cv2
import tempfile
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import time

from utils import load_config, ensure_dir, format_timestamp, setup_logger
from detector import PPEDetector
from video_processor import VideoProcessor, batch_process

logger = setup_logger("safex.app")

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="SafeX - PPE Violation Detection",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #3d3d5c;
    }
    .violation-critical { color: #FF4B4B; font-weight: bold; }
    .violation-high { color: #FF8C00; font-weight: bold; }
    .violation-medium { color: #FFD700; font-weight: bold; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "config": None,
        "detector": None,
        "processor": None,
        "results": None,
        "processing": False,
        "video_path": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def load_system():
    """Load config, detector, and processor."""
    if st.session_state.config is None:
        config_path = Path("config.yaml")
        if config_path.exists():
            st.session_state.config = load_config(str(config_path))
        else:
            st.session_state.config = get_default_config()

    if st.session_state.detector is None:
        with st.spinner("Loading AI model... (first time may download weights)"):
            st.session_state.detector = PPEDetector(st.session_state.config)
            st.session_state.processor = VideoProcessor(
                st.session_state.detector, st.session_state.config
            )


def get_default_config() -> dict:
    """Fallback configuration."""
    return {
        "model": {
            "weights": "yolov8m.pt",
            "confidence_threshold": 0.45,
            "iou_threshold": 0.5,
            "device": "auto",
            "img_size": 640
        },
        "ppe_classes": {
            "helmet": {"present_class": "helmet", "absent_class": "no-helmet"},
            "vest": {"present_class": "vest", "absent_class": "no-vest"},
            "safety_shoes": {"present_class": "safety-shoes", "absent_class": "no-safety-shoes"}
        },
        "person_detection": {"class_id": 0, "min_confidence": 0.5, "min_area": 5000},
        "video": {"frame_sample_fps": 2, "batch_size": 8, "max_resolution": 1080},
        "violation_rules": {
            "required_ppe": ["helmet", "vest", "safety_shoes"],
            "min_violation_duration_sec": 2,
            "cooldown_sec": 30,
            "severity_levels": {
                "critical": ["helmet"],
                "high": ["vest"],
                "medium": ["safety_shoes"]
            }
        },
        "zones": {"enabled": False},
        "output": {
            "save_annotated_frames": True,
            "save_annotated_video": True,
            "save_violation_crops": True,
            "report_format": "csv",
            "output_dir": "./output",
            "violation_snapshots_dir": "./output/violations"
        }
    }


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    """Render sidebar with settings."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/safety-helmet.png", width=60)
        st.title("SafeX Settings")
        st.divider()

        # Model Settings
        st.subheader("\U0001f916 Model")
        model_option = st.selectbox(
            "Model Size",
            ["yolov8n.pt (Fast)", "yolov8s.pt (Balanced)", "yolov8m.pt (Accurate)", "yolov8l.pt (Best)", "Custom"],
            index=2
        )

        if model_option == "Custom":
            custom_path = st.text_input("Custom weights path", placeholder="path/to/best.pt")
            if custom_path:
                st.session_state.config["model"]["weights"] = custom_path
        else:
            st.session_state.config["model"]["weights"] = model_option.split(" ")[0]

        confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.45, 0.05)
        st.session_state.config["model"]["confidence_threshold"] = confidence

        st.divider()

        # Processing Settings
        st.subheader("\u2699\ufe0f Processing")
        sample_fps = st.slider("Analysis FPS", 1, 10, 2, 1,
                              help="Frames per second to analyze. Higher = more thorough, slower.")
        st.session_state.config["video"]["frame_sample_fps"] = sample_fps

        st.divider()

        # PPE Requirements
        st.subheader("\U0001f6e1\ufe0f Required PPE")
        helmet_req = st.checkbox("Helmet", value=True)
        vest_req = st.checkbox("Safety Vest", value=True)
        shoes_req = st.checkbox("Safety Shoes", value=True)

        required = []
        if helmet_req: required.append("helmet")
        if vest_req: required.append("vest")
        if shoes_req: required.append("safety_shoes")
        st.session_state.config["violation_rules"]["required_ppe"] = required

        st.divider()

        # Info
        st.caption("SafeX v1.0 | PPE Violation Detection")
        st.caption("Powered by YOLOv8 + OpenCV")


# ============================================================
# Main Pages
# ============================================================
def render_home():
    """Render the home/upload page."""
    st.markdown('<p class="main-header">\U0001f6e1\ufe0f SafeX</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered PPE Safety Violation Detection for Industrial Plants</p>', unsafe_allow_html=True)
    st.write("")

    # Upload section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("\U0001f4f9 Upload Footage")
        uploaded_files = st.file_uploader(
            "Drop your MP4/AVI/MOV files here",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=True,
            help="Upload plant camera footage for PPE violation analysis"
        )

        # OR specify folder path
        st.write("**Or** specify a folder path with video files:")
        folder_path = st.text_input(
            "Video folder path",
            placeholder="/path/to/your/footage/folder",
            help="Path to a local folder containing MP4 files"
        )

    with col2:
        st.subheader("\u2139\ufe0f Quick Start")
        st.info("""
        **How to use:**
        1. Upload video file(s) or enter folder path
        2. Adjust settings in sidebar
        3. Click 'Start Analysis'
        4. View violations & export report
        """)

        st.metric("Supported Formats", "MP4, AVI, MOV, MKV")
        st.metric("Max Upload", "500 MB")

    st.divider()

    # Start processing
    if uploaded_files or folder_path:
        if st.button("\U0001f680 Start Analysis", type="primary", use_container_width=True):
            process_videos(uploaded_files, folder_path)


def process_videos(uploaded_files, folder_path):
    """Process uploaded or folder-based videos."""
    load_system()

    processor = st.session_state.processor
    all_reports = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            st.subheader(f"\U0001f3ac Processing: {uploaded_file.name}")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_cb(current, total):
                pct = min(current / max(total, 1), 1.0)
                progress_bar.progress(pct)
                status_text.text(f"Frame {current}/{total} ({pct:.0%})")

            report = processor.process_video(tmp_path, progress_callback=progress_cb)
            all_reports.append(report)

            progress_bar.progress(1.0)
            status_text.text("Complete!")
            os.unlink(tmp_path)

    elif folder_path and os.path.isdir(folder_path):
        st.subheader(f"\U0001f4c2 Batch Processing: {folder_path}")
        all_reports = batch_process(folder_path, st.session_state.config)

    if all_reports:
        st.session_state.results = all_reports
        render_results(all_reports)


def render_results(reports: list):
    """Render analysis results and violation dashboard."""
    st.divider()
    st.header("\U0001f4ca Analysis Results")

    # Aggregate metrics
    total_violations = sum(r.get("summary", {}).get("total_violations", 0) for r in reports)
    total_persons = sum(r.get("summary", {}).get("total_persons_detected", 0) for r in reports)
    total_frames = sum(r.get("summary", {}).get("total_frames_processed", 0) for r in reports)
    avg_violation_rate = np.mean([r.get("summary", {}).get("violation_rate", 0) for r in reports])

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("\U0001f6a8 Total Violations", total_violations,
                 delta=None, delta_color="inverse")
    with col2:
        st.metric("\U0001f464 Persons Detected", total_persons)
    with col3:
        st.metric("\U0001f3ac Frames Analyzed", total_frames)
    with col4:
        st.metric("\u26a0\ufe0f Violation Rate", f"{avg_violation_rate:.1f}%")

    st.divider()

    # Violation details
    for report in reports:
        if "error" in report:
            st.error(f"Failed: {report['video_file']} - {report['error']}")
            continue

        st.subheader(f"\U0001f4f9 {report['video_file']}")

        # Severity breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Violation by PPE Type:**")
            violation_types = report.get("violation_by_type", {})
            if violation_types:
                df_types = pd.DataFrame(
                    list(violation_types.items()),
                    columns=["PPE Item", "Count"]
                )
                st.bar_chart(df_types.set_index("PPE Item"))
            else:
                st.success("No violations detected!")

        with col2:
            st.write("**Violation by Severity:**")
            severity = report.get("violation_by_severity", {})
            if any(v > 0 for v in severity.values()):
                df_sev = pd.DataFrame(
                    list(severity.items()),
                    columns=["Severity", "Count"]
                )
                st.bar_chart(df_sev.set_index("Severity"))

        # Violations table
        violations = report.get("violations", [])
        if violations:
            st.write("**Detailed Violations Log:**")
            df = pd.DataFrame(violations)
            df_display = df[["timestamp_fmt", "frame_number", "missing_ppe", "severity", "zone"]].copy()
            df_display.columns = ["Time", "Frame", "Missing PPE", "Severity", "Zone"]
            st.dataframe(df_display, use_container_width=True, height=300)

        # Output files
        if report.get("annotated_video_path"):
            st.write("**\U0001f3ac Annotated Video:**")
            st.info(f"Saved to: {report['annotated_video_path']}")

        if report.get("report_path"):
            st.write("**\U0001f4c4 Report:**")
            st.info(f"Saved to: {report['report_path']}")

    # Export
    st.divider()
    st.subheader("\U0001f4e5 Export")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("\U0001f4c4 Export JSON Report"):
            report_json = json.dumps(reports, indent=2, default=str)
            st.download_button(
                "Download JSON",
                report_json,
                file_name="safex_report.json",
                mime="application/json"
            )
    with col2:
        if st.button("\U0001f4ca Export CSV"):
            all_violations = []
            for r in reports:
                all_violations.extend(r.get("violations", []))
            if all_violations:
                df = pd.DataFrame(all_violations)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv_data,
                    file_name="safex_violations.csv",
                    mime="text/csv"
                )


def render_realtime():
    """Render real-time analysis view with multi-camera RTSP support."""
    st.header("\U0001f534 Live Camera Monitor")

    st.markdown("""
    Connect to **RTSP cameras**, **webcam**, or a **video file** for real-time PPE monitoring.
    Violations are detected live and alerts are sent to configured channels.
    """)

    # Camera source selection
    col1, col2 = st.columns([3, 1])
    with col1:
        source_type = st.radio(
            "Source Type",
            ["RTSP Camera URL", "Webcam (USB)", "Video File (simulate live)"],
            horizontal=True
        )

    # Source input based on type
    if source_type == "RTSP Camera URL":
        source = st.text_input(
            "RTSP URL",
            value="rtsp://admin:password@192.168.1.101:554/stream1",
            help="Get this from your NVR/DVR admin panel. Test with VLC first."
        )
        st.caption("Common formats: Hikvision `rtsp://admin:pass@IP:554/Streaming/Channels/101` | Dahua `rtsp://admin:pass@IP:554/cam/realmonitor?channel=1`")
    elif source_type == "Webcam (USB)":
        cam_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0)
        source = str(cam_index)
    else:
        source = st.text_input("Video File Path", placeholder="/path/to/your/footage.mp4")

    # Camera name for logging
    camera_name = st.text_input("Camera Name (for reports)", value="Camera-1")

    # Alert settings
    with st.expander("\U0001f514 Alert Settings"):
        alert_enabled = st.checkbox("Enable Telegram Alerts", value=False)
        if alert_enabled:
            tg_token = st.text_input("Bot Token", type="password", help="From @BotFather")
            tg_chat = st.text_input("Chat ID", help="Your Telegram group/chat ID")
            cooldown = st.slider("Alert cooldown (seconds)", 10, 300, 60)

    st.divider()

    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        start_btn = st.button("\U0001f534 Start Live Monitor", type="primary", use_container_width=True)
    with col2:
        stop_btn = st.button("\u23f9 Stop", use_container_width=True)
    with col3:
        st.metric("Status", "Ready" if not st.session_state.get("live_running") else "LIVE")

    if start_btn and source:
        load_system()
        run_realtime_enhanced(source, camera_name)


def run_realtime_enhanced(source, camera_name="Camera-1"):
    """Enhanced real-time detection with metrics dashboard."""
    try:
        source_val = int(source)
    except ValueError:
        source_val = source

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        st.error(f"Cannot open video source: {source}")
        st.info("Troubleshooting: \n- Check if the RTSP URL is correct\n- Ensure the camera is on the same network\n- Test the URL in VLC Media Player first")
        return

    # Get stream info
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st.success(f"Connected to {camera_name}: {width}x{height} @ {fps:.0f}fps")

    # Live display area
    frame_placeholder = st.empty()

    # Metrics row
    metric_cols = st.columns(5)
    with metric_cols[0]:
        persons_metric = st.empty()
    with metric_cols[1]:
        violations_metric = st.empty()
    with metric_cols[2]:
        fps_metric = st.empty()
    with metric_cols[3]:
        frames_metric = st.empty()
    with metric_cols[4]:
        total_violations_metric = st.empty()

    # Violation log
    st.subheader("\U0001f4cb Live Violation Log")
    log_placeholder = st.empty()

    detector = st.session_state.detector
    frame_count = 0
    total_violations = 0
    violation_log = []
    process_every_n = max(1, int(fps / st.session_state.config["video"]["frame_sample_fps"]))

    import time as _time
    start_time = _time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # For video files, loop back
            if isinstance(source_val, str) and not source_val.startswith("rtsp"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                st.warning("Stream disconnected. Attempting reconnect...")
                cap.release()
                _time.sleep(3)
                cap = cv2.VideoCapture(source_val)
                continue

        if frame_count % process_every_n == 0:
            timestamp = frame_count / fps
            result = detector.detect_frame(frame, frame_count, timestamp)
            annotated = detector.annotate_frame(frame, result)

            # Display frame
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update metrics
            elapsed = _time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            persons_metric.metric("Persons", len(result.persons))
            violations_metric.metric("Frame Violations", len(result.violations))
            fps_metric.metric("Processing FPS", f"{current_fps:.1f}")
            frames_metric.metric("Frames", frame_count)
            total_violations_metric.metric("Total Violations", total_violations)

            # Log violations
            if result.violations:
                total_violations += len(result.violations)
                for v in result.violations:
                    from utils import format_timestamp
                    violation_log.append({
                        "Time": format_timestamp(timestamp),
                        "Missing PPE": ", ".join(v.missing_ppe),
                        "Severity": v.severity,
                        "Confidence": f"{v.person_bbox.confidence:.0%}"
                    })

                # Update log display (show last 20)
                if violation_log:
                    df_log = pd.DataFrame(violation_log[-20:][::-1])
                    log_placeholder.dataframe(df_log, use_container_width=True)

        frame_count += 1

    cap.release()
    st.info(f"Stream ended. Processed {frame_count} frames, detected {total_violations} violations.")


# ============================================================
# Main App
# ============================================================
def main():
    init_session_state()
    render_sidebar()

    # Navigation
    tab1, tab2, tab3 = st.tabs(["\U0001f3e0 Upload & Analyze", "\U0001f534 Real-Time", "\U0001f4c4 History"])

    with tab1:
        render_home()

    with tab2:
        render_realtime()

    with tab3:
        st.header("\U0001f4c4 Previous Results")
        if st.session_state.results:
            render_results(st.session_state.results)
        else:
            st.info("No previous analysis results. Upload a video to get started.")


if __name__ == "__main__":
    main()
