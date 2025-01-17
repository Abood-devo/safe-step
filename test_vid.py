from ultralytics import YOLO
import cv2
import time
import numpy as np

def process_video_yolo(video_path, output_path, conf_threshold=0.5):
    # Initialize YOLO model
    model = YOLO('yolo11n.pt')
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    start_time = time.time()
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection
        results = model(frame, conf=conf_threshold)[0]
        
        # Draw detections
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            label = f'{model.names[cls]} {conf:.2f}'
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Show progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    out.release()
    
    elapsed_time = time.time() - start_time
    fps_processing = frame_count / elapsed_time
    
    print(f"\nProcessing Complete:")
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"Average processing speed: {fps_processing:.2f} FPS")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process video with YOLO')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('output_path', type=str, help='Path to save processed video')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()
    
    process_video_yolo(args.video_path, args.output_path, args.conf)