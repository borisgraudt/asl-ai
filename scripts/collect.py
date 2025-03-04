import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def create_directory_structure():
    """Создание структуры директорий для данных"""
    base_dir = "data/raw_gestures"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Создаем папки для каждой буквы (A-Z)
    for letter in range(ord('A'), ord('Z')+1):
        letter_dir = os.path.join(base_dir, chr(letter))
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)
    
    return base_dir

def collect_data():
    """Сбор данных жестов"""
    base_dir = create_directory_structure()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return
    
    current_letter = 'A'
    frame_count = 0
    recording = False
    target_frames = 400  # Увеличиваем целевое количество кадров
    min_confidence = 0.7  # Минимальная уверенность для сохранения кадра
    frames_without_hand = 0
    MAX_FRAMES_WITHOUT_HAND = 10  # Автоматическая остановка записи после 10 кадров без руки
    
    # Настройка окна
    cv2.namedWindow('Data Collection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Data Collection', 1280, 720)
    
    print("\n=== ASL Gesture Data Collection ===")
    print("\nИнструкции:")
    print("- Нажмите 'r' для начала/остановки записи")
    print("- Нажмите 'n' для перехода к следующей букве")
    print("- Нажмите 'b' для возврата к предыдущей букве")
    print("- Нажмите 'd' для удаления последних 10 кадров")
    print("- Нажмите 'q' для выхода")
    print(f"- Целевое количество кадров для каждой буквы: {target_frames}")
    print("\nСоветы:")
    print("- Держите руку в центре кадра")
    print("- Меняйте положение и угол руки для разнообразия данных")
    print("- Запись автоматически остановится, если рука не видна")
    print(f"\nТекущая буква: {current_letter}")
    
    last_save_time = time.time()
    MIN_SAVE_INTERVAL = 0.1  # Минимальный интервал между сохранениями (100 мс)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Отражаем кадр для естественного отображения
        frame = cv2.flip(frame, 1)
        
        # Обработка кадра
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Создаем информационную панель
        info_panel = np.zeros((150, frame.shape[1], 3), dtype=np.uint8)
        
        # Отображение информации
        cv2.putText(info_panel, f"Letter: {current_letter}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(info_panel, f"{'Recording' if recording else 'Waiting'}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if recording else (200, 200, 200), 2)
        cv2.putText(info_panel, f"Frames: {frame_count}/{target_frames}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Добавляем полосу прогресса
        progress = int((frame_count / target_frames) * frame.shape[1])
        cv2.rectangle(info_panel, (0, 130), (progress, 140), (0, 255, 0), -1)
        cv2.rectangle(info_panel, (0, 130), (frame.shape[1], 140), (255, 255, 255), 2)
        
        hand_detected = False
        if results.multi_hand_landmarks:
            hand_detected = True
            frames_without_hand = 0
            
            # Отрисовка руки
            for hand_landmarks in results.multi_hand_landmarks:
                # Рисуем подсветку руки
                hand_mask = np.zeros_like(frame)
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_point = hand_landmarks.landmark[start_idx]
                    end_point = hand_landmarks.landmark[end_idx]
                    
                    start_pos = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
                    end_pos = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
                    
                    cv2.line(hand_mask, start_pos, end_pos, (255, 255, 255), 15)
                
                # Размытие маски
                hand_mask = cv2.GaussianBlur(hand_mask, (25, 25), 0)
                
                # Наложение подсветки
                frame = cv2.addWeighted(frame, 1, hand_mask, 0.3, 0)
                
                # Отрисовка ключевых точек
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(50, 205, 50), thickness=3, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 140, 0), thickness=5, circle_radius=5)
                )
                
                # Сохранение данных при записи
                if recording and time.time() - last_save_time >= MIN_SAVE_INTERVAL:
                    # Проверяем, что рука полностью в кадре
                    all_points_visible = all(0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0 
                                          for landmark in hand_landmarks.landmark)
                    
                    if all_points_visible:
                        points = []
                        for landmark in hand_landmarks.landmark:
                            points.extend([landmark.x, landmark.y, landmark.z])
                        
                        filename = os.path.join(base_dir, current_letter, f"frame_{frame_count}.npy")
                        np.save(filename, np.array(points))
                        frame_count += 1
                        last_save_time = time.time()
                        
                        # Добавляем индикатор записи
                        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        
        if not hand_detected and recording:
            frames_without_hand += 1
            if frames_without_hand >= MAX_FRAMES_WITHOUT_HAND:
                recording = False
                print("\nЗапись автоматически остановлена: рука не обнаружена")
        
        # Объединяем кадр и информационную панель
        combined_frame = np.vstack([frame, info_panel])
        cv2.imshow('Data Collection', combined_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            frames_without_hand = 0
            if recording:
                print(f"\nНачало записи буквы {current_letter}")
            else:
                print(f"\nОстановка записи. Сохранено {frame_count} кадров для буквы {current_letter} (цель: {target_frames})")
        elif key == ord('n'):
            if frame_count < target_frames:
                print(f"\nВнимание: записано только {frame_count} кадров из {target_frames}.")
                print("Уверены, что хотите перейти к следующей букве? (y/n)")
                confirm = cv2.waitKey(0)
                if confirm != ord('y'):
                    continue
            recording = False
            frame_count = 0
            if ord(current_letter) < ord('Z'):
                current_letter = chr(ord(current_letter) + 1)
                print(f"\nПереход к букве: {current_letter}")
            else:
                print("\nВы достигли последней буквы (Z)")
        elif key == ord('b'):
            if ord(current_letter) > ord('A'):
                recording = False
                frame_count = 0
                current_letter = chr(ord(current_letter) - 1)
                print(f"\nВозврат к букве: {current_letter}")
        elif key == ord('d'):
            # Удаление последних 10 кадров
            if frame_count > 0:
                for i in range(min(10, frame_count)):
                    filename = os.path.join(base_dir, current_letter, f"frame_{frame_count-1}.npy")
                    if os.path.exists(filename):
                        os.remove(filename)
                    frame_count -= 1
                print(f"\nУдалено 10 последних кадров. Текущий count: {frame_count}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data() 