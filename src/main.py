import cv2
import numpy as np
from utils.camera import Camera
from detectors.face_detector import FaceDetector
from utils.visuals import Visualizer
import time

def main():
    # Initialize components
    camera = Camera(camera_id=0, width=1280, height=720)  # Lower resolution for testing
    face_detector = FaceDetector()
    visualizer = Visualizer()

    # Start camera
    if not camera.start():
        print("Failed to start camera")
        return

    print("Camera started successfully. Press 'q' to quit.")

    try:
        while True:
            # Read frame
            ret, frame = camera.read()
            if not ret:
                print("Failed to read frame")
                continue

            # Process frame
            faces = face_detector.process(frame)
            
            # Get stats
            stats = camera.get_stats()
            
            # Draw results
            frame = visualizer.draw_results(frame, faces=faces, stats=stats)
            
            # Display frame
            cv2.imshow('Face Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Main loop error: {str(e)}")

    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()