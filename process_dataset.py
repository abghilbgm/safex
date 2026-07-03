import cv2
import os
from pathlib import Path
from ultralytics import YOLO

def process_videos(video_dir: str, output_dir: str):
    """
    Process videos to separate frames with human movement from those without.
    This helps in training models for human/motion detection.
    """
    model = YOLO("yolov8n.pt")  # Use YOLOv8 nano for fast inference
    
    video_dir_path = Path(video_dir)
    video_paths = list(video_dir_path.glob("*.mp4"))
    
    if not video_paths:
        print(f"No .mp4 files found in {video_dir_path}")
        return

    human_dir = Path(output_dir) / "human"
    background_dir = Path(output_dir) / "background"
    
    human_dir.mkdir(parents=True, exist_ok=True)
    background_dir.mkdir(parents=True, exist_ok=True)
    
    for video_path in video_paths:
        print(f"Processing video: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Failed to open {video_path.name}")
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25  # Fallback
            
        frame_count = 0
        saved_human_count = 0
        saved_bg_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process 1 frame per second to avoid saving too many identical frames
            if frame_count % fps == 0:
                results = model(frame, verbose=False)
                has_human = False
                
                for result in results:
                    for box in result.boxes:
                        label = result.names[int(box.cls[0])]
                        if label == "person":
                            has_human = True
                            break
                    if has_human:
                        break
                
                timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                filename = f"{video_path.stem}_ms{timestamp}.jpg"
                
                if has_human:
                    cv2.imwrite(str(human_dir / filename), frame)
                    saved_human_count += 1
                else:
                    cv2.imwrite(str(background_dir / filename), frame)
                    saved_bg_count += 1
                    
            frame_count += 1
            
        cap.release()
        print(f"Finished {video_path.name}: {saved_human_count} human frames, {saved_bg_count} background frames saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from dataset based on human presence")
    parser.add_argument("--input", type=str, default="Data_Train", help="Input directory containing mp4 videos")
    parser.add_argument("--output", type=str, default="extracted_dataset", help="Output directory for saved frames")
    args = parser.parse_args()
    
    process_videos(args.input, args.output)
