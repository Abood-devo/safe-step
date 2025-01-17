import cv2
import time
from datetime import datetime

def test_camera(camera_id=0):
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera Properties:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print("\nControls:")
    print("Press 's' to save image")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Display frame
        cv2.imshow('Camera Test', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Save image when 's' is pressed
        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_test_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved image: {filename}")
        
        # Quit when 'q' is pressed
        elif key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    test_camera()