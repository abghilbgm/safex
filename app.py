"""
SafeX - Real-Time PPE Safety Monitor v2.0
Live camera feed with violation detection, unique IDs, zoom, and instant alerts.
No file uploads — built for always-on industrial camera monitoring.

Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from utils import load_config, ensure_dir, setup_logger
from detector import PPEDetector
from alerts import AlertManager

logger = setup_logger("safex.app")

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="SafeX — Live PPE Monitor",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header { font-size:2.2rem; font-weight:700; color:#FF4B4B; text-align:center; }
    .sub-header { font-size:0.9rem; color:#888; text-align:center; margin-bottom:1rem; }
    .live-badge {
        display:inline-block; background:#FF4B4B; color:white;
        padding:3px 12px; border-radius:12px; font-size:0.8rem; font-weight:bold;
        animation: pulse 1.5s infinite;
    }
    .offline-badge {
        display:inline-block; background:#444; color:#ccc;
        padding:3px 12px; border-radius:12px; font-size:0.8rem;
    }
    .violation-card {
        background:#1e0f0f; border:1px solid #FF4B4B33;
        border-left: 3px solid #FF4B4B;
        border-radius:6px; padding:10px 12px; margin-bottom:8px;
        font-size:0.85rem;
    }
    .violation-high { border-left-color: #FF8C00 !important; background:#1e150a !important; }
    .violation-medium { border-left-color: #FFD700 !important; background:#1a190a !important; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
    .stMetric [data-testid="metric-container"] { background:#1a1a2e; border-radius:8px; padding:10px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
def init_state():
    """Initialize all session state variables."""
    defaults = {
        "config": None,
        "detector": None,
        "alert_manager": None,
        "running": False,
        "monitor": None,
        "latest_frame": None,
        "latest_raw_frame": None,
        "violations_log": [],
        "violation_count": 0,
        "persons_count": 0,
        "vio_id_counter": 0,
        "camera_status": "\u26ab Offline",
        "last_alert_text": "",
        "fps_display": 0.0,
        "zoom_factor": 1.0,
        "zoom_x": 50,
        "zoom_y": 50,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def load_system():
    """Load config, detector, and alert manager (once per session)."""
    if st.session_state.config is None:
        p = Path("config.yaml")
        st.session_state.config = load_config(str(p)) if p.exists() else _default_config()

    if st.session_state.detector is None:
        with st.spinner("Loading YOLOv8 model\u2026"):
            st.session_state.detector = PPEDetector(st.session_state.config)

    if st.session_state.alert_manager is None:
        st.session_state.alert_manager = AlertManager(st.session_state.config)


def _default_config() -> dict:
    """Fallback configuration when config.yaml is not found."""
    return {
        "model": {
            "weights": "yolov8n.pt",
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
        "video": {"frame_sample_fps": 5, "batch_size": 1, "max_resolution": 720},
        "violation_rules": {
            "required_ppe": ["helmet", "vest"],
            "min_violation_duration_sec": 1,
            "cooldown_sec": 15,
            "severity_levels": {
                "critical": ["helmet"],
                "high": ["vest"],
                "medium": ["safety_shoes"]
            }
        },
        "zones": {"enabled": False},
        "alerts": {
            "enabled": False,
            "cooldown_seconds": 30,
            "telegram": {"enabled": False},
            "email": {"enabled": False},
            "webhook": {"enabled": False}
        },
        "output": {
            "save_annotated_frames": False,
            "save_annotated_video": False,
            "save_violation_crops": True,
            "output_dir": "./output",
            "violation_snapshots_dir": "./output/violations"
        }
    }


# ============================================================
# Background Live Monitor
# ============================================================
class LiveMonitor:
    """
    Background thread: reads camera frames, runs PPE detection,
    stores results in st.session_state for the Streamlit UI to consume.
    """

    def __init__(
        self,
        source: str,
        detector: PPEDetector,
        alert_manager: AlertManager,
        config: dict,
        process_fps: int = 5
    ):
        self.source = source
        self.detector = detector
        self.alert_manager = alert_manager
        self.config = config
        self.process_fps = process_fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        ensure_dir("./output/violations")

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="safex-live"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    # ----------------------------------------------------------
    def _run(self):
        """Main capture + detection loop."""
        try:
            src = int(self.source)          # webcam index
        except (ValueError, TypeError):
            src = self.source               # RTSP URL or file path

        cap = cv2.VideoCapture(src)
        if isinstance(src, str) and "rtsp" in src.lower():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            st.session_state.camera_status = "\u274c Failed to connect"
            return

        st.session_state.camera_status = "\U0001f7e2 Connected"
        interval = 1.0 / max(self.process_fps, 1)
        last_process = 0.0
        frame_counter = 0
        fps_window: list = []

        while self._running:
            ret, frame = cap.read()
            if not ret:
                st.session_state.camera_status = "\u26a0\ufe0f Stream interrupted"
                time.sleep(1)
                continue

            now = time.time()
            # Track display FPS
            fps_window.append(now)
            fps_window = [t for t in fps_window if now - t < 1.0]
            st.session_state.fps_display = float(len(fps_window))

            # Store raw frame for zoom reference
            st.session_state.latest_raw_frame = frame.copy()

            # Run detection at configured rate
            if now - last_process >= interval:
                last_process = now
                frame_counter += 1

                result = self.detector.detect_frame(
                    frame, frame_number=frame_counter, timestamp=now
                )

                # Annotate frame
                annotated = self.detector.annotate_frame(frame.copy(), result)
                st.session_state.latest_frame = annotated
                st.session_state.persons_count = len(result.persons)

                # Handle violations
                for violation in result.violations:
                    self._log_violation(violation, frame)
                    if self.alert_manager and self.alert_manager.enabled:
                        snapshot = violation.snapshot_path or ""
                        self.alert_manager.send_violation_alert(
                            violation=violation,
                            camera_name="Live",
                            snapshot_path=snapshot
                        )

            time.sleep(0.02)

        cap.release()
        st.session_state.camera_status = "\u26ab Stopped"

    # ----------------------------------------------------------
    def _log_violation(self, violation, frame: np.ndarray):
        """Assign a unique violation ID, save crop, append to log."""
        st.session_state.vio_id_counter += 1
        ts_str = datetime.now().strftime("%H%M%S")
        vio_id = f"VIO-{ts_str}-{st.session_state.vio_id_counter:04d}"

        # Save violation crop
        crop_path = ""
        try:
            h, w = frame.shape[:2]
            bb = violation.person_bbox
            x1, y1 = max(int(bb.x1) - 10, 0), max(int(bb.y1) - 10, 0)
            x2, y2 = min(int(bb.x2) + 10, w), min(int(bb.y2) + 10, h)
            crop = frame[y1:y2, x1:x2]
            crop_path = f"./output/violations/{vio_id}.jpg"
            cv2.imwrite(crop_path, crop)
            violation.snapshot_path = crop_path
        except Exception:
            pass

        entry = {
            "id": vio_id,
            "time": datetime.now().strftime("%H:%M:%S"),
            "severity": violation.severity,
            "missing_ppe": ", ".join(violation.missing_ppe),
            "zone": violation.zone or "General",
            "confidence": f"{violation.person_bbox.confidence:.0%}",
            "snapshot": crop_path,
        }

        # Newest first, keep last 200
        st.session_state.violations_log = (
            [entry] + st.session_state.violations_log
        )[:200]
        st.session_state.violation_count += 1
        st.session_state.last_alert_text = (
            f"\u26a0\ufe0f [{entry['severity'].upper()}] {entry['missing_ppe']} "
            f"\u2014 {entry['time']}  |  ID: {vio_id}"
        )


# ============================================================
# Zoom helper
# ============================================================
def apply_zoom(
    frame: np.ndarray,
    zoom_factor: float,
    cx_pct: float,
    cy_pct: float
) -> np.ndarray:
    """Crop to (cx_pct, cy_pct) center at zoom_factor, then scale back."""
    if zoom_factor <= 1.05:
        return frame
    h, w = frame.shape[:2]
    new_w = max(int(w / zoom_factor), 16)
    new_h = max(int(h / zoom_factor), 16)
    cx = int(w * cx_pct / 100)
    cy = int(h * cy_pct / 100)
    x1 = max(cx - new_w // 2, 0)
    y1 = max(cy - new_h // 2, 0)
    x2 = min(x1 + new_w, w)
    y2 = min(y1 + new_h, h)
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ============================================================
# Sidebar
# ============================================================
def render_sidebar() -> dict:
    """Render settings sidebar. Returns a params dict."""
    with st.sidebar:
        st.markdown("## \U0001f6e1\ufe0f SafeX")
        st.caption("Real-Time PPE Monitor v2.0")
        st.divider()

        # Camera source
        st.subheader("\U0001f4f7 Camera Source")
        source_type = st.radio(
            "Input type",
            ["RTSP Stream", "Webcam", "Video File (test)"],
            horizontal=True
        )
        if source_type == "RTSP Stream":
            cam_url = st.text_input(
                "RTSP URL",
                placeholder="rtsp://user:pass@192.168.1.x:554/stream1"
            )
        elif source_type == "Webcam":
            cam_idx = st.number_input(
                "Webcam index", min_value=0, max_value=10, value=0, step=1
            )
            cam_url = str(int(cam_idx))
        else:
            cam_url = st.text_input(
                "Video file path",
                placeholder="/path/to/test_footage.mp4"
            )

        process_fps = st.slider(
            "Detection FPS", 1, 15, 5, 1,
            help="Frames per second to run inference on."
        )
        st.divider()

        # Model
        st.subheader("\U0001f916 Model")
        model_choice = st.selectbox(
            "Weights",
            ["yolov8n.pt (Fastest)", "yolov8s.pt (Balanced)",
             "yolov8m.pt (Accurate)", "yolov8l.pt (Best)", "Custom"]
        )
        if model_choice == "Custom":
            weights = st.text_input(
                "Custom weights path", placeholder="path/to/best.pt"
            )
        else:
            weights = model_choice.split(" ")[0]
        confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.45, 0.05)
        st.divider()

        # PPE rules
        st.subheader("\U0001f9ba Required PPE")
        helmet_req = st.checkbox("Helmet", value=True)
        vest_req = st.checkbox("Safety Vest", value=True)
        shoes_req = st.checkbox("Safety Shoes", value=False)
        required_ppe = [
            p for p, v in [
                ("helmet", helmet_req),
                ("vest", vest_req),
                ("safety_shoes", shoes_req)
            ] if v
        ]
        st.divider()

        # Alerts
        st.subheader("\U0001f514 Alerts")
        tg_enabled = st.checkbox("Telegram", value=False)
        tg_token, tg_chat = "", ""
        if tg_enabled:
            tg_token = st.text_input("Bot Token", type="password")
            tg_chat = st.text_input("Chat ID")

        wh_enabled = st.checkbox("Webhook (Slack / Teams)", value=False)
        wh_url = ""
        if wh_enabled:
            wh_url = st.text_input(
                "Webhook URL", placeholder="https://hooks.slack.com/..."
            )
        st.divider()
        st.caption("Powered by YOLOv8 + OpenCV")

        return {
            "cam_url": cam_url,
            "process_fps": process_fps,
            "weights": weights,
            "confidence": confidence,
            "required_ppe": required_ppe,
            "tg_enabled": tg_enabled,
            "tg_token": tg_token,
            "tg_chat": tg_chat,
            "wh_enabled": wh_enabled,
            "wh_url": wh_url,
        }


# ============================================================
# Main UI
# ============================================================
def main():
    """Main real-time PPE monitoring dashboard."""
    params = render_sidebar()

    # ── Header ──
    st.markdown(
        '<p class="main-header">\U0001f6e1\ufe0f SafeX \u2014 Live PPE Monitor</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Real-time AI-powered PPE compliance monitoring — industrial cameras</p>',
        unsafe_allow_html=True
    )

    # ── Control Bar ──
    c1, c2, c3, c4 = st.columns([1.2, 1, 2, 1])
    with c1:
        start_btn = st.button(
            "\u25b6\ufe0f Start Monitor",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.running
        )
    with c2:
        stop_btn = st.button(
            "\u23f9 Stop",
            use_container_width=True,
            disabled=not st.session_state.running
        )
    with c3:
        st.markdown(f"**Status:** {st.session_state.camera_status}")
    with c4:
        st.markdown(f"**{st.session_state.fps_display:.0f} FPS**")

    # ── Start Logic ──
    if start_btn:
        cam_url = params["cam_url"]
        if not cam_url and cam_url != "0":
            st.warning("Configure a camera source in the sidebar first.")
        else:
            if st.session_state.config is None:
                st.session_state.config = _default_config()

            # Patch config from sidebar values
            st.session_state.config["model"]["weights"] = params["weights"]
            st.session_state.config["model"]["confidence_threshold"] = params["confidence"]
            st.session_state.config["violation_rules"]["required_ppe"] = params["required_ppe"]

            # Telegram
            if params["tg_enabled"] and params["tg_token"]:
                st.session_state.config.setdefault("alerts", {})
                st.session_state.config["alerts"]["enabled"] = True
                st.session_state.config["alerts"]["telegram"] = {
                    "enabled": True,
                    "bot_token": params["tg_token"],
                    "chat_id": params["tg_chat"],
                }
            # Webhook
            if params["wh_enabled"] and params["wh_url"]:
                st.session_state.config.setdefault("alerts", {})
                st.session_state.config["alerts"]["enabled"] = True
                st.session_state.config["alerts"]["webhook"] = {
                    "enabled": True,
                    "url": params["wh_url"],
                }

            # Reset alert manager so it picks up new config
            st.session_state.alert_manager = None
            load_system()

            monitor = LiveMonitor(
                source=cam_url,
                detector=st.session_state.detector,
                alert_manager=st.session_state.alert_manager,
                config=st.session_state.config,
                process_fps=params["process_fps"],
            )
            st.session_state.monitor = monitor
            st.session_state.running = True
            monitor.start()
            st.rerun()

    # ── Stop Logic ──
    if stop_btn:
        if st.session_state.monitor:
            st.session_state.monitor.stop()
        st.session_state.running = False
        st.session_state.camera_status = "\u26ab Stopped"
        st.rerun()

    st.divider()

    # ── KPI Row ──
    sev_counts = {"critical": 0, "high": 0, "medium": 0}
    for v in st.session_state.violations_log:
        sev = v.get("severity", "medium")
        sev_counts[sev] = sev_counts.get(sev, 0) + 1

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("\U0001f6a8 Total Violations", st.session_state.violation_count)
    with k2:
        st.metric("\U0001f534 Critical", sev_counts["critical"])
    with k3:
        st.metric("\U0001f7e0 High", sev_counts["high"])
    with k4:
        st.metric("\U0001f465 Persons Detected", st.session_state.persons_count)

    st.divider()

    # ── Video Feed + Violation Log (side by side) ──
    video_col, log_col = st.columns([3, 2])

    with video_col:
        # Live / Offline badge
        if st.session_state.running:
            st.markdown('<span class="live-badge">\u25cf LIVE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="offline-badge">\u25cf OFFLINE</span>', unsafe_allow_html=True)

        # Zoom controls
        zc1, zc2, zc3 = st.columns(3)
        with zc1:
            zoom_factor = st.slider(
                "\U0001f50d Zoom", 1.0, 8.0,
                st.session_state.zoom_factor, 0.25,
                key="zoom_factor_slider"
            )
            st.session_state.zoom_factor = zoom_factor
        with zc2:
            zoom_x = st.slider(
                "\u2194\ufe0f Center X%", 5, 95,
                st.session_state.zoom_x, 5,
                key="zoom_x_slider"
            )
            st.session_state.zoom_x = zoom_x
        with zc3:
            zoom_y = st.slider(
                "\u2195\ufe0f Center Y%", 5, 95,
                st.session_state.zoom_y, 5,
                key="zoom_y_slider"
            )
            st.session_state.zoom_y = zoom_y

        # Frame display
        frame_slot = st.empty()
        if st.session_state.latest_frame is not None:
            frame = st.session_state.latest_frame.copy()
            if zoom_factor > 1.05:
                frame = apply_zoom(frame, zoom_factor, zoom_x, zoom_y)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(frame_rgb, use_container_width=True)
        else:
            frame_slot.info(
                "\u23f3 Waiting for camera feed\u2026  "
                "Set a camera source in the sidebar then click **\u25b6\ufe0f Start Monitor**."
            )

        # Last alert banner
        if st.session_state.last_alert_text:
            st.warning(st.session_state.last_alert_text)

    with log_col:
        st.subheader("\U0001f4cb Violation Log")

        lcol1, lcol2 = st.columns(2)
        with lcol1:
            if st.button("\U0001f5d1\ufe0f Clear Log", use_container_width=True):
                st.session_state.violations_log = []
                st.session_state.violation_count = 0
                st.session_state.last_alert_text = ""
                st.rerun()
        with lcol2:
            import pandas as pd, json as _json
            if st.session_state.violations_log:
                csv_bytes = pd.DataFrame(st.session_state.violations_log).drop(
                    columns=["snapshot"], errors="ignore"
                ).to_csv(index=False).encode()
                st.download_button(
                    "\U0001f4e5 Export CSV",
                    csv_bytes,
                    file_name=f"safex_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        if st.session_state.violations_log:
            for entry in st.session_state.violations_log[:25]:
                sev = entry["severity"]
                color = (
                    "#FF4B4B" if sev == "critical"
                    else "#FF8C00" if sev == "high"
                    else "#FFD700"
                )
                extra_class = (
                    "" if sev == "critical"
                    else "violation-high" if sev == "high"
                    else "violation-medium"
                )
                st.markdown(
                    f"""<div class="violation-card {extra_class}">
  <span style="color:{color};font-weight:bold">&#9632; {sev.upper()}</span>
  <span style="float:right;color:#888;font-size:0.8rem">{entry['time']}</span><br>
  <b>ID:</b> <code>{entry['id']}</code><br>
  <b>Missing PPE:</b> {entry['missing_ppe']}<br>
  <b>Zone:</b> {entry.get('zone','General')} &nbsp; <b>Conf:</b> {entry.get('confidence','')}
</div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.success("\u2705 No violations recorded yet.")

    # ── Auto-refresh while live ──
    if st.session_state.running:
        time.sleep(0.12)
        st.rerun()


if __name__ == "__main__":
    main()

