import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

def prepare_data():

    print("Начало подготовки данных...")
    
    data_dir = "data/raw_gestures"
    if not os.path.exists(data_dir):
        print(f"Ошибка: Директория {data_dir} не существует")
        return
    
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Сбор данных
    X = []
    y = []
    
    print("Загрузка и обработка файлов...")
    for letter in os.listdir(data_dir):
        letter_dir = os.path.join(data_dir, letter)
        if not os.path.isdir(letter_dir):
            continue
        
        print(f"Обработка буквы {letter}...")
        for file in os.listdir(letter_dir):
            if file.endswith(".npy"):
                file_path = os.path.join(letter_dir, file)
                coordinates = np.load(file_path)
                # --- Position-invariant preprocessing ---
                # coordinates: (63,) -> (21, 3)
                coords = coordinates.reshape(-1, 3)
                palm = coords[0]  # landmark 0
                coords_centered = coords - palm  # центрирование
                max_dist = np.linalg.norm(coords_centered, axis=1).max()
                if max_dist > 0:
                    coords_centered /= max_dist  # масштабирование
                coordinates = coords_centered.flatten()
                # ---
                X.append(coordinates)
                y.append(letter)
    
    # Преобразование в numpy массивы
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nРазмерность данных: {X.shape}")
    print(f"Количество классов: {len(np.unique(y))}")
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Нормализация данных
    print("\nНормализация данных...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Кодирование меток
    print("Кодирование меток...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Сохранение обработанных данных
    print("\nСохранение обработанных данных...")
    np.save("data/processed/X_train.npy", X_train_scaled)
    np.save("data/processed/X_test.npy", X_test_scaled)
    np.save("data/processed/y_train.npy", y_train_encoded)
    np.save("data/processed/y_test.npy", y_test_encoded)
    
    # Сохранение scaler и encoder
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    print("\nСтатистика:")
    print(f"Размер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер тестовой выборки: {X_test.shape[0]}")
    print(f"Количество признаков: {X_train.shape[1]}")
    print(f"Классы: {le.classes_}")
    
    print("\nПодготовка данных завершена!")

if __name__ == "__main__":
    prepare_data() 