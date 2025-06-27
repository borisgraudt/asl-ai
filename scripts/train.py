import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Lambda # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import pennylane as qml
from pennylane import numpy as pnp
from tensorflow.keras import Input, Model # type: ignore
import time

def to_float32(x):
    return tf.cast(x, tf.float32)

def create_quantum_layer(n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev, interface='tf')
    def circuit(inputs, weights1, weights2):
        # inputs: shape (1, n_qubits)
        for i in range(n_qubits):
            qml.RX(inputs[0][i], wires=i)
        # Первый слой параметров
        for i in range(n_qubits):
            qml.RY(weights1[i], wires=i)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
        # Второй слой параметров
        for i in range(n_qubits):
            qml.RY(weights2[i], wires=i)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    weight_shapes = {"weights1": n_qubits, "weights2": n_qubits}
    return qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=n_qubits)

def create_model(input_shape, num_classes, batch_size=64):
    from tensorflow.keras import Input, Model # type: ignore
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
    from tensorflow.keras.optimizers import Adam # type: ignore
    inp = Input(shape=(input_shape,), batch_size=batch_size)
    x = Dense(256, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_training(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Точность на обучении')
    plt.plot(history['val_accuracy'], label='Точность на валидации')
    plt.title('Динамика точности')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Потери на обучении')
    plt.plot(history['val_loss'], label='Потери на валидации')
    plt.title('Динамика функции потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.show()

def train_model():
    """Обучение модели"""
    print("Загрузка данных...")
    
    # Загрузка данных
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")
    
    # Загрузка label encoder для получения количества классов
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    
    batch_size = 64
    # Обрезаем до кратного batch_size (можно не обрезать, но для единообразия)
    n_train = (X_train.shape[0] // batch_size) * batch_size
    n_test = (X_test.shape[0] // batch_size) * batch_size
    X_train, y_train = X_train[:n_train], y_train[:n_train]
    X_test, y_test = X_test[:n_test], y_test[:n_test]
    
    print("\nСоздание модели...")
    model = create_model(X_train.shape[1], len(le.classes_), batch_size=batch_size)
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\nНачало обучения...")
    start_time = time.time()
    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - start_time
    print(f"\nВремя обучения: {elapsed/60:.2f} минут")
    
    plot_training(hist.history)
    
    # Оценка модели
    print("\nОценка модели...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0, batch_size=1)
    print(f"Точность на тестовой выборке: {accuracy*100:.2f}%")
    
    # Матрица ошибок
    y_pred = model.predict(X_test, batch_size=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    # Детальный отчет
    print("\nДетальный отчет по классификации:")
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_))
    
    # Сохранение модели
    print("\nСохранение модели...")
    model.save('models/model.keras')
    print("Обучение завершено!")

if __name__ == "__main__":
    train_model() 