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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# World-class CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"], .main {
    background-color: #07080d !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #c8cdd8;
}
[data-testid="stSidebar"] {
    background-color: #0b0d18 !important;
    border-right: 1px solid #181c2e !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.4rem 1rem 1rem; }
.block-container { padding-top: 1.2rem !important; max-width: 100% !important; }

/* ── Header ── */
.sx-header {
    display: flex; align-items: center; justify-content: center;
    gap: 12px; padding: 20px 0 4px;
}
.sx-shield { color: #e63946; line-height: 1; }
.sx-title {
    font-size: 1.9rem; font-weight: 700; letter-spacing: -0.04em;
    color: #edf2f7; margin: 0; line-height: 1;
}
.sx-title b { color: #e63946; font-weight: 800; }
.sx-sub {
    text-align: center; font-size: 0.72rem; color: #2e3556;
    letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 1.4rem;
}

/* ── Status strip ── */
.sx-status {
    display: flex; align-items: center; gap: 28px;
    background: #0b0d18; border: 1px solid #181c2e;
    border-radius: 8px; padding: 9px 18px; margin-bottom: 14px;
    font-size: 0.8rem;
}
.sx-status-item { display: flex; align-items: center; gap: 6px; }
.sx-dot {
    width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
}
.sx-dot-live   { background:#e63946; box-shadow:0 0 7px #e6394688; animation: sx-blink 1.2s ease infinite; }
.sx-dot-idle   { background:#2a2e45; }
.sx-dot-ok     { background:#2dd4bf; box-shadow:0 0 7px #2dd4bf66; }
.sx-dot-warn   { background:#f59e0b; }
.sx-dot-err    { background:#e63946; }
.sx-lbl  { font-size:0.68rem; color:#2e3556; text-transform:uppercase; letter-spacing:0.1em; }
.sx-val  { color:#9ba3bf; font-weight:500; }
.sx-fps  { font-family:'JetBrains Mono',monospace; color:#4fc3f7; font-size:0.8rem; font-weight:500; }
.sx-sep  { width:1px; height:18px; background:#181c2e; }
@keyframes sx-blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* ── KPI cards ── */
.sx-kpi-row { display:flex; gap:10px; margin-bottom:14px; }
.sx-kpi {
    flex:1; background:#0b0d18;
    border:1px solid #181c2e; border-radius:10px;
    padding:14px 18px; position:relative; overflow:hidden;
}
.sx-kpi::after {
    content:''; position:absolute;
    top:0; left:0; right:0; height:2px;
    background: var(--accent);
}
.sx-kpi-lbl { font-size:0.66rem; color:#2e3556; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px; }
.sx-kpi-val {
    font-family:'JetBrains Mono',monospace;
    font-size:2.1rem; font-weight:700; line-height:1;
    color: var(--accent);
}

/* ── Control buttons ── */
.stButton > button {
    font-family:'Inter',sans-serif !important;
    font-size:0.78rem !important;
    font-weight:600 !important;
    letter-spacing:0.06em !important;
    text-transform:uppercase !important;
    border-radius:6px !important;
    padding:9px 20px !important;
    transition:all 0.15s ease !important;
}
.stButton > button[kind="primary"] {
    background:#e63946 !important; color:#fff !important; border:none !important;
}
.stButton > button[kind="primary"]:hover {
    background:#c8303a !important; box-shadow:0 4px 16px #e6394440 !important;
    transform:translateY(-1px) !important;
}
.stButton > button:not([kind="primary"]) {
    background:#0f1120 !important; color:#6b7594 !important;
    border:1px solid #1e2340 !important;
}
.stButton > button:not([kind="primary"]):hover {
    background:#141829 !important; color:#9ba3bf !important;
    border-color:#2a3060 !important;
}
.stButton > button:disabled {
    opacity:0.3 !important; cursor:not-allowed !important;
}

/* ── Feed area ── */
.sx-feed-wrap {
    border:1px solid #181c2e; border-radius:10px; overflow:hidden;
    background:#07080d; margin-bottom:8px;
}
.sx-feed-offline {
    height:340px; display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    background:#0b0d18; border:1px dashed #181c2e;
    border-radius:10px; gap:10px;
}
.sx-feed-offline-icon { color:#1e2235; }
.sx-feed-offline-text { font-size:0.8rem; color:#2a2e45; letter-spacing:0.08em; }

/* ── Zoom strip ── */
.sx-zoom-wrap {
    background:#0b0d18; border:1px solid #181c2e;
    border-radius:8px; padding:10px 14px 4px; margin-top:6px;
}
.sx-zoom-label {
    font-size:0.65rem; color:#2e3556;
    text-transform:uppercase; letter-spacing:0.12em;
    margin-bottom:2px;
}

/* ── Alert banner ── */
.sx-alert {
    background:#140808; border:1px solid #e6394430;
    border-left:3px solid #e63946;
    border-radius:0 6px 6px 0; padding:8px 14px;
    font-size:0.79rem; color:#e63946;
    font-family:'JetBrains Mono',monospace;
    margin-top:8px; line-height:1.5;
}

/* ── Violation log ── */
.sx-log-header {
    display:flex; justify-content:space-between;
    align-items:center; margin-bottom:10px;
}
.sx-log-title { font-size:0.68rem; color:#2e3556; text-transform:uppercase; letter-spacing:0.14em; }
.sx-log-count { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#2e3556; }
.sx-vcard {
    border-left:3px solid; border-radius:0 6px 6px 0;
    padding:9px 13px; margin-bottom:5px;
    background:#0b0d18;
    border-top:1px solid #181c2e;
    border-right:1px solid #181c2e;
    border-bottom:1px solid #181c2e;
    font-size:0.8rem; line-height:1.65;
}
.sx-vcard.c { border-left-color:#e63946; }
.sx-vcard.h { border-left-color:#f59e0b; }
.sx-vcard.m { border-left-color:#f0b429; }
.sx-vid  { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#3a4268; }
.sx-vsev { font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; }
.sx-vsev.c { color:#e63946; } .sx-vsev.h { color:#f59e0b; } .sx-vsev.m { color:#f0b429; }
.sx-vppe { color:#9ba3bf; font-size:0.82rem; font-weight:500; }
.sx-vmeta { color:#3a4268; font-size:0.72rem; }
.sx-vtime { float:right; font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#2e3556; }
.sx-empty { text-align:center; padding:32px 0; font-size:0.78rem; color:#1e2235; letter-spacing:0.08em; }

/* ── Sidebar styles ── */
.sx-sb-title { font-size:1rem; font-weight:700; color:#edf2f7; letter-spacing:-0.02em; line-height:1; margin:0; }
.sx-sb-title span { color:#e63946; }
.sx-sb-ver { font-size:0.65rem; color:#1e2235; letter-spacing:0.12em; text-transform:uppercase; margin-top:2px; }
.sx-sb-sec { font-size:0.63rem; font-weight:600; color:#2e3556; text-transform:uppercase; letter-spacing:0.14em; margin:18px 0 8px; padding-left:2px; border-left:2px solid #181c2e; padding-left:8px; }
.sx-divider { border:none; border-top:1px solid #181c2e; margin:14px 0; }

/* ── Streamlit widget overrides ── */
div[data-testid="stSlider"] label p,
div[data-testid="stTextInput"] label p,
div[data-testid="stSelectbox"] label p,
div[data-testid="stRadio"] label p,
div[data-testid="stNumberInput"] label p,
div[data-testid="stCheckbox"] label p {
    font-size:0.73rem !important; color:#4a5272 !important;
    text-transform:uppercase; letter-spacing:0.09em;
}
div[data-testid="stTextInput"] input {
    background:#0b0d18 !important; border:1px solid #181c2e !important;
    color:#c8cdd8 !important; border-radius:6px !important;
    font-size:0.82rem !important; font-family:'Inter',sans-serif !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color:#e63946 !important; box-shadow:0 0 0 1px #e6394430 !important;
    outline:none !important;
}
div[data-testid="stSelectbox"] > div > div {
    background:#0b0d18 !important; border:1px solid #181c2e !important;
    border-radius:6px !important; color:#c8cdd8 !important;
}
[data-testid="stRadio"] div[role="radio"] {
    font-size:0.8rem !important; color:#6b7594 !important;
}
div[data-testid="stSlider"] [data-testid="stSliderThumb"] { background:#e63946 !important; }
div[data-testid="stSlider"] div[role="slider"] { border-color:#e63946 !important; }
.stSlider [data-baseweb="slider"] div[data-testid="stSlider"] div { background:#e63946 !important; }
div[data-testid="stCheckbox"] svg { color:#e63946 !important; }
[data-testid="stMetric"],
[data-testid="metric-container"] { display:none !important; }
hr { border-color:#181c2e !important; }
.stSpinner > div { border-top-color:#e63946 !important; }
p, li { color:#9ba3bf !important; }
code { background:#0f1120 !important; border:1px solid #181c2e !important; color:#4fc3f7 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#07080d; }
::-webkit-scrollbar-thumb { background:#1e2340; border-radius:2px; }
::-webkit-scrollbar-thumb:hover { background:#2e3556; }

/* ── Force dark on the full app shell ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: #07080d !important;
}
[data-testid="stHeader"] { background-color: #07080d !important; border-bottom: 1px solid #181c2e !important; }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background-color: #07080d !important; }

/* ── Hide Streamlit branding footer ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── BaseWeb popover / dropdown (selectbox open state) ── */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] > div > div {
    background-color: #0b0d18 !important;
    border: 1px solid #181c2e !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 32px #00000088 !important;
}
[data-baseweb="menu"] {
    background-color: #0b0d18 !important;
}
[data-baseweb="list-item"],
[role="option"] {
    background-color: #0b0d18 !important;
    color: #9ba3bf !important;
    font-size: 0.82rem !important;
}
[data-baseweb="list-item"]:hover,
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background-color: #141829 !important;
    color: #f0f2f7 !important;
}

/* ── BaseWeb input / select ── */
[data-baseweb="base-input"],
[data-baseweb="input"],
[data-baseweb="textarea"] {
    background-color: #0b0d18 !important;
    border-color: #181c2e !important;
    color: #c8cdd8 !important;
}
[data-baseweb="select"] > div {
    background-color: #0b0d18 !important;
    border-color: #181c2e !important;
    color: #c8cdd8 !important;
}
[data-baseweb="select"] > div:hover,
[data-baseweb="select"] > div:focus-within {
    border-color: #e63946 !important;
    box-shadow: 0 0 0 1px #e6394430 !important;
}

/* ── BaseWeb tooltip ── */
[data-baseweb="tooltip"] > div {
    background-color: #141829 !important;
    color: #9ba3bf !important;
    border: 1px solid #181c2e !important;
    border-radius: 6px !important;
    font-size: 0.75rem !important;
}

/* ── Radio buttons ── */
[data-testid="stRadio"] > div {
    gap: 6px;
}
[data-testid="stRadio"] label {
    background-color: #0b0d18 !important;
    border: 1px solid #181c2e !important;
    border-radius: 6px !important;
    padding: 4px 12px !important;
    cursor: pointer !important;
    transition: border-color 0.15s !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    border-color: #e63946 !important;
    background-color: #140810 !important;
}

/* ── Slider track / thumb ── */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
    background-color: #181c2e !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #e63946 !important;
    border-color: #e63946 !important;
}

/* ── Number input ── */
[data-testid="stNumberInput"] input {
    background-color: #0b0d18 !important;
    border: 1px solid #181c2e !important;
    color: #c8cdd8 !important;
    border-radius: 6px !important;
}
[data-testid="stNumberInput"] button {
    background-color: #0f1120 !important;
    border-color: #181c2e !important;
    color: #6b7594 !important;
}

/* ── Notification / warning / info overrides ── */
[data-testid="stNotification"] {
    background-color: #0b0d18 !important;
    border-color: #181c2e !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background-color: #0f1120 !important;
    color: #6b7594 !important;
    border: 1px solid #1e2340 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #141829 !important;
    color: #9ba3bf !important;
    border-color: #2a3060 !important;
}
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
        "camera_status": "offline",
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
            st.session_state.camera_status = "error"
            return

        st.session_state.camera_status = "connected"
        interval = 1.0 / max(self.process_fps, 1)
        last_process = 0.0
        frame_counter = 0
        fps_window: list = []

        while self._running:
            ret, frame = cap.read()
            if not ret:
                st.session_state.camera_status = "interrupted"
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
        st.session_state.camera_status = "offline"

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
            f"[{entry['severity'].upper()}]  {entry['missing_ppe']}  —  {entry['time']}  |  {vio_id}"
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
        st.markdown(
            '<p class="sx-sb-title"><b>Safe</b>X</p>'
            '<p class="sx-sb-ver">Real-Time PPE Monitor &nbsp;v2.0</p>',
            unsafe_allow_html=True
        )
        st.markdown('<hr class="sx-divider">', unsafe_allow_html=True)

        # ─ Camera source ─
        st.markdown('<div class="sx-sb-sec">Camera Source</div>', unsafe_allow_html=True)
        source_type = st.radio(
            "Input type",
            ["RTSP Stream", "Webcam", "Video File"],
            horizontal=True, label_visibility="collapsed"
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
                "File path",
                placeholder="/path/to/footage.mp4"
            )

        process_fps = st.slider("Detection FPS", 1, 15, 5, 1)
        st.markdown('<hr class="sx-divider">', unsafe_allow_html=True)

        # ─ Model ─
        st.markdown('<div class="sx-sb-sec">Detection Model</div>', unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Weights",
            ["yolov8n.pt  —  Fastest", "yolov8s.pt  —  Balanced",
             "yolov8m.pt  —  Accurate", "yolov8l.pt  —  Best", "Custom"]
        )
        if model_choice == "Custom":
            weights = st.text_input("Weights path", placeholder="path/to/best.pt")
        else:
            weights = model_choice.split("  ")[0]
        confidence = st.slider("Confidence", 0.1, 0.9, 0.45, 0.05)
        st.markdown('<hr class="sx-divider">', unsafe_allow_html=True)

        # ─ PPE rules ─
        st.markdown('<div class="sx-sb-sec">Required PPE</div>', unsafe_allow_html=True)
        helmet_req = st.checkbox("Helmet", value=True)
        vest_req   = st.checkbox("Safety Vest", value=True)
        shoes_req  = st.checkbox("Safety Shoes", value=False)
        required_ppe = [
            p for p, v in [
                ("helmet", helmet_req),
                ("vest", vest_req),
                ("safety_shoes", shoes_req)
            ] if v
        ]
        st.markdown('<hr class="sx-divider">', unsafe_allow_html=True)

        # ─ Alerts ─
        st.markdown('<div class="sx-sb-sec">Alerts</div>', unsafe_allow_html=True)
        tg_enabled = st.checkbox("Telegram", value=False)
        tg_token, tg_chat = "", ""
        if tg_enabled:
            tg_token = st.text_input("Bot Token", type="password")
            tg_chat  = st.text_input("Chat ID")

        wh_enabled = st.checkbox("Webhook  (Slack / Teams)", value=False)
        wh_url = ""
        if wh_enabled:
            wh_url = st.text_input("URL", placeholder="https://hooks.slack.com/...")
        st.markdown('<hr class="sx-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.62rem;color:#1e2235;letter-spacing:0.1em;">'
            'POWERED BY YOLOV8 + OPENCV</p>',
            unsafe_allow_html=True
        )

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
def _status_dot(status: str) -> str:
    """Return a CSS dot class based on camera status string."""
    m = {
        "connected":   "sx-dot-ok",
        "interrupted": "sx-dot-warn",
        "error":       "sx-dot-err",
    }
    return m.get(status, "sx-dot-idle")


def _status_label(status: str) -> str:
    m = {
        "connected":   "Connected",
        "interrupted": "Interrupted",
        "error":       "Error",
        "offline":     "Offline",
    }
    return m.get(status, status.capitalize())


def main():
    """Main real-time PPE monitoring dashboard."""
    params = render_sidebar()

    # ── Header (one shield, no other icons) ──
    SHIELD_SVG = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        'fill="#e63946" width="36" height="36">'
        '<path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 '
        '9-12V5l-9-4z"/>'
        '</svg>'
    )
    st.markdown(
        f'<div class="sx-header">{SHIELD_SVG}'
        f'<h1 class="sx-title"><b>Safe</b>X &mdash; Live PPE Monitor</h1></div>'
        f'<p class="sx-sub">AI-Powered Industrial Safety Compliance &nbsp;&bull;&nbsp; Real-Time Detection</p>',
        unsafe_allow_html=True
    )

    # ── Control row ──
    c1, c2, c_space = st.columns([1.1, 0.9, 4])
    with c1:
        start_btn = st.button(
            "Start Monitor", type="primary",
            use_container_width=True, disabled=st.session_state.running
        )
    with c2:
        stop_btn = st.button(
            "Stop", use_container_width=True,
            disabled=not st.session_state.running
        )

    # Status strip
    live_dot  = "sx-dot-live" if st.session_state.running else "sx-dot-idle"
    live_text = "Live" if st.session_state.running else "Idle"
    cam_dot   = _status_dot(st.session_state.camera_status)
    cam_text  = _status_label(st.session_state.camera_status)
    fps_val   = f"{st.session_state.fps_display:.0f}"
    st.markdown(
        f'<div class="sx-status">'
        f'  <div class="sx-status-item">'
        f'    <span class="sx-dot {live_dot}"></span>'
        f'    <span class="sx-lbl">Monitor&nbsp;</span>'
        f'    <span class="sx-val">{live_text}</span>'
        f'  </div>'
        f'  <div class="sx-sep"></div>'
        f'  <div class="sx-status-item">'
        f'    <span class="sx-dot {cam_dot}"></span>'
        f'    <span class="sx-lbl">Camera&nbsp;</span>'
        f'    <span class="sx-val">{cam_text}</span>'
        f'  </div>'
        f'  <div class="sx-sep"></div>'
        f'  <div class="sx-status-item">'
        f'    <span class="sx-lbl">FPS&nbsp;</span>'
        f'    <span class="sx-fps">{fps_val}</span>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Start logic ──
    if start_btn:
        cam_url = params["cam_url"]
        if not cam_url and cam_url != "0":
            st.warning("Configure a camera source in the sidebar first.")
        else:
            if st.session_state.config is None:
                st.session_state.config = _default_config()
            st.session_state.config["model"]["weights"] = params["weights"]
            st.session_state.config["model"]["confidence_threshold"] = params["confidence"]
            st.session_state.config["violation_rules"]["required_ppe"] = params["required_ppe"]
            if params["tg_enabled"] and params["tg_token"]:
                st.session_state.config.setdefault("alerts", {})
                st.session_state.config["alerts"]["enabled"] = True
                st.session_state.config["alerts"]["telegram"] = {
                    "enabled": True, "bot_token": params["tg_token"],
                    "chat_id": params["tg_chat"],
                }
            if params["wh_enabled"] and params["wh_url"]:
                st.session_state.config.setdefault("alerts", {})
                st.session_state.config["alerts"]["enabled"] = True
                st.session_state.config["alerts"]["webhook"] = {
                    "enabled": True, "url": params["wh_url"],
                }
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

    # ── Stop logic ──
    if stop_btn:
        if st.session_state.monitor:
            st.session_state.monitor.stop()
        st.session_state.running = False
        st.session_state.camera_status = "offline"
        st.rerun()

    # ── KPI cards (pure HTML) ──
    sev_counts = {"critical": 0, "high": 0, "medium": 0}
    for v in st.session_state.violations_log:
        sev_counts[v.get("severity", "medium")] = sev_counts.get(v.get("severity", "medium"), 0) + 1

    st.markdown(
        f'<div class="sx-kpi-row">'
        f'  <div class="sx-kpi sx-kpi-total"   style="--accent:#e63946">'
        f'    <div class="sx-kpi-lbl">Total Violations</div>'
        f'    <div class="sx-kpi-val">{st.session_state.violation_count}</div>'
        f'  </div>'
        f'  <div class="sx-kpi sx-kpi-critical" style="--accent:#e63946">'
        f'    <div class="sx-kpi-lbl">Critical</div>'
        f'    <div class="sx-kpi-val">{sev_counts["critical"]}</div>'
        f'  </div>'
        f'  <div class="sx-kpi sx-kpi-high"     style="--accent:#f59e0b">'
        f'    <div class="sx-kpi-lbl">High</div>'
        f'    <div class="sx-kpi-val">{sev_counts["high"]}</div>'
        f'  </div>'
        f'  <div class="sx-kpi sx-kpi-persons"  style="--accent:#2dd4bf">'
        f'    <div class="sx-kpi-lbl">Persons Detected</div>'
        f'    <div class="sx-kpi-val">{st.session_state.persons_count}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Two-column layout: video feed | violation log ──
    video_col, log_col = st.columns([3, 2], gap="medium")

    with video_col:
        # Zoom controls
        st.markdown('<div class="sx-zoom-wrap">', unsafe_allow_html=True)
        zc1, zc2, zc3 = st.columns(3)
        with zc1:
            st.markdown('<div class="sx-zoom-label">Zoom</div>', unsafe_allow_html=True)
            zoom_factor = st.slider(
                "z", 1.0, 8.0, st.session_state.zoom_factor, 0.25,
                key="zoom_factor_slider", label_visibility="collapsed"
            )
            st.session_state.zoom_factor = zoom_factor
        with zc2:
            st.markdown('<div class="sx-zoom-label">Center X %</div>', unsafe_allow_html=True)
            zoom_x = st.slider(
                "x", 5, 95, st.session_state.zoom_x, 5,
                key="zoom_x_slider", label_visibility="collapsed"
            )
            st.session_state.zoom_x = zoom_x
        with zc3:
            st.markdown('<div class="sx-zoom-label">Center Y %</div>', unsafe_allow_html=True)
            zoom_y = st.slider(
                "y", 5, 95, st.session_state.zoom_y, 5,
                key="zoom_y_slider", label_visibility="collapsed"
            )
            st.session_state.zoom_y = zoom_y
        st.markdown('</div>', unsafe_allow_html=True)

        # Video frame
        frame_slot = st.empty()
        if st.session_state.latest_frame is not None:
            frame = st.session_state.latest_frame.copy()
            if zoom_factor > 1.05:
                frame = apply_zoom(frame, zoom_factor, zoom_x, zoom_y)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(frame_rgb, use_container_width=True)
        else:
            frame_slot.markdown(
                '<div class="sx-feed-offline">'
                '  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
                'fill="#1e2235" width="40" height="40">'
                '<path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 '
                '1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/>'
                '  </svg>'
                '  <span class="sx-feed-offline-text">No signal &mdash; configure camera source and start monitor</span>'
                '</div>',
                unsafe_allow_html=True
            )

        # Alert banner
        if st.session_state.last_alert_text:
            st.markdown(
                f'<div class="sx-alert">{st.session_state.last_alert_text}</div>',
                unsafe_allow_html=True
            )

    with log_col:
        n = len(st.session_state.violations_log)
        st.markdown(
            f'<div class="sx-log-header">'
            f'  <span class="sx-log-title">Violation Log</span>'
            f'  <span class="sx-log-count">{n} events</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        act1, act2 = st.columns(2)
        with act1:
            if st.button("Clear Log", use_container_width=True):
                st.session_state.violations_log = []
                st.session_state.violation_count = 0
                st.session_state.last_alert_text = ""
                st.rerun()
        with act2:
            import pandas as pd
            if st.session_state.violations_log:
                csv_bytes = pd.DataFrame(st.session_state.violations_log).drop(
                    columns=["snapshot"], errors="ignore"
                ).to_csv(index=False).encode()
                st.download_button(
                    "Export CSV", csv_bytes,
                    file_name=f"safex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", use_container_width=True,
                )

        if st.session_state.violations_log:
            log_html = ""
            for entry in st.session_state.violations_log[:30]:
                sev = entry["severity"]
                s = "c" if sev == "critical" else "h" if sev == "high" else "m"
                log_html += (
                    f'<div class="sx-vcard {s}">'
                    f'  <span class="sx-vtime">{entry["time"]}</span>'
                    f'  <span class="sx-vsev {s}">{sev.upper()}</span><br>'
                    f'  <span class="sx-vppe">{entry["missing_ppe"]}</span><br>'
                    f'  <span class="sx-vmeta">'
                    f'    Zone: {entry.get("zone","General")} &nbsp;&bull;&nbsp; '
                    f'    Conf: {entry.get("confidence","")}'
                    f'  </span><br>'
                    f'  <span class="sx-vid">{entry["id"]}</span>'
                    f'</div>'
                )
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="sx-empty">No violations recorded</div>',
                unsafe_allow_html=True
            )

    # ── Auto-refresh while live ──
    if st.session_state.running:
        time.sleep(0.12)
        st.rerun()


if __name__ == "__main__":
    main()

