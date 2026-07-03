# SafeX - PPE Safety Violation Detection System

> Industry-grade AI-powered PPE (Personal Protective Equipment) compliance monitoring for alumina refinery and industrial plant camera footage.

## Features

- **Real-time & Batch Processing** - Analyze live camera feeds or uploaded MP4 footage
- **Multi-PPE Detection** - Helmet, Safety Vest, Safety Shoes violation detection
- **YOLOv8 Powered** - State-of-the-art object detection with fine-tuning support
- **Zone-based Rules** - Define restricted areas with specific PPE requirements
- **Severity Classification** - Critical/High/Medium violation categorization
- **Annotated Output** - Generated video with bounding boxes and violation labels
- **Comprehensive Reports** - CSV/JSON reports with timestamps, snapshots, and statistics
- **Streamlit Dashboard** - Professional web UI for upload, analysis, and monitoring
- **Telegram Alerts** - Real-time violation notifications (configurable)

## Project Structure

```
safex/
|-- app.py                 # Streamlit web application (main entry point)
|-- detector.py            # YOLOv8 PPE detection engine
|-- video_processor.py     # Video frame extraction & batch processing
|-- utils.py               # Data classes, geometry, logging utilities
|-- config.yaml            # Configuration (model, rules, zones, output)
|-- requirements.txt       # Python dependencies
|-- README.md              # This file
|-- output/                # Generated results (auto-created)
|   |-- violations/        # Violation snapshot crops
|   |-- *_annotated.mp4    # Annotated output videos
|   |-- *_report.csv       # Violation reports
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

This launches the web dashboard at `http://localhost:8501`

### 3. Analyze Footage

**Option A: Web UI (Recommended)**
- Open the Streamlit app in browser
- Upload your MP4 files or specify a folder path
- Adjust detection settings in the sidebar
- Click "Start Analysis"

**Option B: Command Line (Batch)**
```python
from utils import load_config
from detector import PPEDetector
from video_processor import VideoProcessor, batch_process

config = load_config("config.yaml")
reports = batch_process("/path/to/your/videos", config)
```

**Option C: Single Video (Python)**
```python
config = load_config("config.yaml")
detector = PPEDetector(config)
processor = VideoProcessor(detector, config)
report = processor.process_video("/path/to/video.mp4")
print(f"Violations found: {report['summary']['total_violations']}")
```

## Model Options

| Model | Speed | Accuracy | Use Case |
| --- | --- | --- | --- |
| yolov8n.pt | Fastest | Good | Real-time / low-resource |
| yolov8s.pt | Fast | Better | Balanced |
| yolov8m.pt | Medium | High | Recommended default |
| yolov8l.pt | Slow | Best | Offline batch processing |
| Custom (fine-tuned) | Varies | Highest | Production deployment |

### Fine-Tuning for Your Plant

For best results, fine-tune on your own plant footage:

1. **Label data** - Use [Roboflow](https://roboflow.com) or [CVAT](https://cvat.ai) to annotate frames with: `helmet`, `no-helmet`, `vest`, `no-vest`, `safety-shoes`, `no-safety-shoes`
2. **Export in YOLO format**
3. **Fine-tune:**
```python
from ultralytics import YOLO
model = YOLO("yolov8m.pt")
model.train(data="your_ppe_dataset.yaml", epochs=100, imgsz=640)
```
4. **Update config.yaml** with path to `best.pt`

## Configuration

Edit `config.yaml` to customize:
- Model weights and confidence thresholds
- Required PPE items per zone
- Severity levels for different violations
- Frame sampling rate
- Output formats and directories

## Requirements

- Python 3.10+
- 8GB+ RAM (16GB recommended)
- GPU optional but recommended for large videos (NVIDIA CUDA)
- Webcam/RTSP for real-time mode

## Roadmap

- [x] Core detection engine (YOLOv8)
- [x] Batch video processing
- [x] Streamlit dashboard
- [x] Violation reporting (CSV/JSON)
- [ ] Fine-tuned PPE model weights
- [ ] Person re-identification tracking
- [ ] RTSP live stream integration
- [ ] Telegram/Email alert system
- [ ] Historical analytics dashboard
- [ ] Edge deployment (ONNX/TensorRT)
