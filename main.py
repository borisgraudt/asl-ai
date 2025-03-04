import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model # type: ignore

def load_resources():

    print("Загрузка модели и необходимых файлов...")
    try:
        model = load_model('models/model.keras')
        
        with open("models/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            
        print("Модель и файлы успешно загружены!")
        return model, le, scaler
    except Exception as e:
        print(f"Ошибка при загрузке файлов: {e}")
        return None, None, None

def process_landmarks(landmarks, scaler):

    # Извлекаем координаты точек
    points = []
    for landmark in landmarks.landmark:
        points.extend([landmark.x, landmark.y, landmark.z])
    
    # Преобразуем в numpy массив
    features = np.array(points)
    
    # Нормализация
    features = scaler.transform(features.reshape(1, -1))
    
    return features

def main():
    # Загрузка ресурсов
    model, le, scaler = load_resources()
    if None in (model, le, scaler):
        return
    
    # Инициализация MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Инициализация камеры
    print("Инициализация камеры...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return
    
    # Настройка окна
    cv2.namedWindow('ASL Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ASL Recognition', 1280, 720)
    
    # Переменные для сглаживания предсказаний
    prediction_history = []
    HISTORY_LENGTH = 5
    
    print("Готово! Нажмите 'q' для выхода.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось получить кадр с камеры")
            break
        
        # Отражаем кадр для естественного отображения
        frame = cv2.flip(frame, 1)
        
        # Обработка кадра
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Создаем информационную панель
        info_display = np.zeros((200, frame.shape[1], 3), dtype=np.uint8)
        
        if results.multi_hand_landmarks:
            # Отрисовка руки
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    # Стиль для соединений
                    mp_drawing.DrawingSpec(
                        color=(50, 205, 50),  # Светло-зеленый
                        thickness=3,
                        circle_radius=1
                    ),
                    # Стиль для ключевых точек
                    mp_drawing.DrawingSpec(
                        color=(255, 140, 0),  # Оранжевый для суставов
                        thickness=5,
                        circle_radius=5
                    )
                )
                
                # Добавляем подсветку контура руки
                for landmark in landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 15, (224, 255, 255), -1)  # Мягкая подсветка
                
                try:
                    # Обработка ключевых точек
                    features = process_landmarks(landmarks, scaler)
                    
                    # Предсказание
                    prediction = model.predict(features, verbose=0)
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[0][predicted_class]
                    
                    # Добавляем предсказание в историю
                    prediction_history.append(predicted_class)
                    if len(prediction_history) > HISTORY_LENGTH:
                        prediction_history.pop(0)
                    
                    # Используем самое частое предсказание
                    from collections import Counter
                    most_common = Counter(prediction_history).most_common(1)[0]
                    predicted_class = most_common[0]
                    stability = most_common[1] / len(prediction_history)
                    
                    # Получаем букву
                    predicted_letter = le.inverse_transform([predicted_class])[0]
                    
                    # Отображение информации
                    cv2.putText(info_display, f"Letter: {predicted_letter}", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    cv2.putText(info_display, f"Confidence: {confidence*100:.1f}%", (20, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(info_display, f"Stability: {stability*100:.1f}%", (20, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                except Exception as e:
                    print(f"Ошибка при обработке кадра: {e}")
        else:
            # Если рука не обнаружена
            cv2.putText(info_display, "Show the gesture to the camera", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Объединяем основной кадр и информационную панель
        combined_frame = np.vstack([frame, info_display])
        
        # Показываем результат
        cv2.imshow('ASL Recognition', combined_frame)
        
        # Выход по клавише 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождение ресурсов
    print("Завершение работы...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
