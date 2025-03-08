import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

def create_model(input_shape, num_classes):
    """Создание архитектуры модели для распознавания букв"""
    model = Sequential([
        # Входной слой
        Dense(2048, input_shape=(input_shape,), activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Скрытые слои
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Выходной слой (26 букв английского алфавита)
        Dense(26, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.00005),  # Уменьшаем learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Обучение модели"""
    print("Загрузка данных...")
    
    # Загрузка данных
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")
    
    # Нормализация данных
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    # Создаем маппинг для букв (A=0, B=1, ...)
    letters = [chr(i) for i in range(65, 91)]  # A-Z
    label_mapping = {i: letter for i, letter in enumerate(letters)}
    
    # Сохраняем label encoder
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_mapping, f)
    
    # Сохраняем scaler
    scaler_params = {'mean': mean, 'std': std}
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler_params, f)
    
    print("\nСоздание модели...")
    model = create_model(X_train.shape[1], len(letters))
    model.summary()
    
    # Улучшенные callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,  # Увеличиваем patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,  # Более агрессивное уменьшение learning rate
            patience=10,
            min_lr=1e-8,
            verbose=1
        )
    ]
    
    print("\nНачало обучения...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=150,  # Увеличиваем количество эпох
        batch_size=16,  # Уменьшаем размер батча
        callbacks=callbacks,
        verbose=1
    )
    
    # Создание директории для графиков
    os.makedirs("plots", exist_ok=True)
    
    # График обучения
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.title('Динамика точности')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Потери на обучении')
    plt.plot(history.history['val_loss'], label='Потери на валидации')
    plt.title('Динамика функции потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.close()
    
    # Оценка модели
    print("\nОценка модели...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Точность на тестовой выборке: {accuracy*100:.2f}%")
    
    # Матрица ошибок
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    plt.figure(figsize=(15, 15))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=letters,
                yticklabels=letters)
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    # Детальный отчет
    print("\nДетальный отчет по классификации:")
    print(classification_report(y_test, y_pred_classes, target_names=letters))
    
    # Сохранение модели
    print("\nСохранение модели...")
    model.save('models/model.keras')
    print("Обучение завершено!")

if __name__ == "__main__":
    train_model() 