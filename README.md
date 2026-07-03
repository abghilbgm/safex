# SAFEX

Safety monitoring and alerting system for alumina refinery PPE and safety-violation detection.

## What this workspace contains
- RTSP video ingestion and snapshot capture
- PPE/safety-violation inference hooks
- Telegram alert delivery for violations
- Training-data layout for normal vs violation footage

## Suggested stack
- Python 3.11+
- OpenCV + Ultralytics YOLOv8-style detection
- python-telegram-bot
- RTSP camera input

## Quick start
1. Activate the environment:
   source .venv/bin/activate
2. Install dependencies:
   pip install -r requirements.txt
3. Fill in your values in .env (the file is already created for you)
4. Run the app:
   python -m app.main
   or
   ./run_safex.sh

## Next milestones
- Add your RTSP URL and Telegram bot credentials in .env
- Train or integrate a PPE/helmet/vest detection model
- Add person ID tracking for UAT IDs such as m1, m2
- Connect snapshot saving and Telegram alert forwarding
