import cv2
import mediapipe as mp
import numpy as np
import os
import time
from pathlib import Path

def create_directories():
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def process_landmarks(hand_landmarks):
    if not hand_landmarks:
        return None
    
    # Extract coordinates for all hand landmarks
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    coords = np.array(features, dtype=np.float32).reshape(-1, 3)
    palm = coords[0]
    coords_centered = coords - palm
    max_dist = np.linalg.norm(coords_centered, axis=1).max()
    if max_dist > 0:
        coords_centered /= max_dist
    return coords_centered.flatten()

def collect_gesture_data(gesture_name, num_samples=400, fps=30):
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    
    # Create output directories
    data_dir = create_directories()
    gesture_dir = data_dir / gesture_name
    gesture_dir.mkdir(exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    print(f"\nStarting recording for gesture '{gesture_name}'")
    print("Press SPACE to start recording")
    print("Press SPACE again to stop recording")
    print("Press 'q' to quit")
    
    recording = False
    frame_count = 0
    sequence = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame")
            break
            
        # Convert frame for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display status
        status = "Recording..." if recording else "Waiting..."
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Recorded: {len(sequence)}/{num_samples}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            if not recording:
                recording = True
                sequence = []
                frame_count = 0
                print("Starting recording...")
            else:
                recording = False
                print("Stopping recording...")
        
        # Save samples
        if recording and results.multi_hand_landmarks:
            features = process_landmarks(results.multi_hand_landmarks[0])
            if features is not None:
                sequence.append(features)
                frame_count += 1
                
                if frame_count >= 30:  # Record 30 frames per sample
                    if len(sequence) < num_samples:
                        # Save sequence
                        sequence_array = np.array(sequence)
                        np.save(gesture_dir / f"gesture_{len(sequence)}.npy", sequence_array)
                        sequence = []
                        frame_count = 0
                        print(f"Saved {len(sequence)}/{num_samples} samples")
                    else:
                        recording = False
                        print("Reached maximum number of samples")
        
        cv2.imshow("Gesture Collection", frame)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nRecording for gesture '{gesture_name}' finished")

def main():
    print("Gesture recording utility")
    print("=" * 50)
    
    while True:
        gesture_name = input("\nEnter gesture name (or 'q' to quit): ").strip()
        
        if gesture_name.lower() == 'q':
            break
            
        if not gesture_name:
            print("Gesture name cannot be empty")
            continue
            
        collect_gesture_data(gesture_name)

if __name__ == "__main__":
    main() 