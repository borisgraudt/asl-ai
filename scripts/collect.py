import cv2
import mediapipe as mp
import numpy as np
import os
import time
from pathlib import Path

def create_directories():
    """Создание необходимых директорий"""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def process_landmarks(hand_landmarks):
    """Обработка точек руки в массив признаков"""
    if not hand_landmarks:
        return None
    
    # Извлекаем координаты всех точек руки
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(features, dtype=np.float32)

def collect_gesture_data(gesture_name, num_samples=400, fps=30):
    """Сбор данных для динамического жеста"""
    # Инициализация MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    
    # Создание директорий
    data_dir = create_directories()
    gesture_dir = data_dir / gesture_name
    gesture_dir.mkdir(exist_ok=True)
    
    # Инициализация камеры
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    print(f"\nНачинаем запись жеста '{gesture_name}'")
    print(f"Нажмите 'SPACE' для начала записи")
    print(f"Нажмите 'SPACE' снова для остановки записи")
    print(f"Нажмите 'q' для выхода")
    
    recording = False
    frame_count = 0
    sequence = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра")
            break
            
        # Конвертируем изображение для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Отрисовка точек руки
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Отображение статуса
        status = "Recording..." if recording else "Waiting..."
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Recorded: {len(sequence)}/{num_samples}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Обработка клавиш
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
        
        # Запись данных
        if recording and results.multi_hand_landmarks:
            features = process_landmarks(results.multi_hand_landmarks[0])
            if features is not None:
                sequence.append(features)
                frame_count += 1
                
                if frame_count >= 30:  # Записываем 30 кадров для каждого жеста
                    if len(sequence) < num_samples:
                        # Сохраняем последовательность
                        sequence_array = np.array(sequence)
                        np.save(gesture_dir / f"gesture_{len(sequence)}.npy", sequence_array)
                        sequence = []
                        frame_count = 0
                        print(f"Записано {len(sequence)}/{num_samples} жестов")
                    else:
                        recording = False
                        print("Достигнуто максимальное количество жестов")
        
        cv2.imshow("Gesture Collection", frame)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nЗапись жеста '{gesture_name}' завершена")

def main():
    """Основная функция"""
    print("Программа для записи динамических жестов")
    print("=" * 50)
    
    while True:
        gesture_name = input("\nВведите название жеста (или 'q' для выхода): ").strip()
        
        if gesture_name.lower() == 'q':
            break
            
        if not gesture_name:
            print("Название жеста не может быть пустым")
            continue
            
        collect_gesture_data(gesture_name)

if __name__ == "__main__":
    main() 