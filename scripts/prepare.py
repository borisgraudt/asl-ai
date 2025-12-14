import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

def prepare_data():

    print("Starting data preparation...")
    
    # For a lightweight repo, we ship only a small sample dataset by default.
    # Point this to your full dataset folder if available.
    data_dir = os.getenv("ASL_AI_RAW_DATA_DIR", "data/sample_raw_gestures")
    if not os.path.exists(data_dir):
        print(f"Error: directory {data_dir} does not exist")
        return
    
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Collect samples
    X = []
    y = []
    
    print("Loading and processing files...")
    for letter in os.listdir(data_dir):
        letter_dir = os.path.join(data_dir, letter)
        if not os.path.isdir(letter_dir):
            continue
        
        print(f"Processing letter {letter}...")
        for file in os.listdir(letter_dir):
            if file.endswith(".npy"):
                file_path = os.path.join(letter_dir, file)
                coordinates = np.load(file_path)
                # --- Position-invariant preprocessing ---
                # coordinates: (63,) -> (21, 3)
                coords = coordinates.reshape(-1, 3)
                palm = coords[0]  # landmark 0
                coords_centered = coords - palm  # centering
                max_dist = np.linalg.norm(coords_centered, axis=1).max()
                if max_dist > 0:
                    coords_centered /= max_dist  # scaling
                coordinates = coords_centered.flatten()
                # ---
                X.append(coordinates)
                y.append(letter)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Encode labels
    print("Encoding labels...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Save processed arrays (UNSCALED features; scaling happens in training)
    print("\nSaving processed data...")
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train_encoded)
    np.save("data/processed/y_test.npy", y_test_encoded)
    
    # Save label encoder (scaler is saved during training)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    print("\nStats:")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Num features: {X_train.shape[1]}")
    print(f"Classes: {list(le.classes_)}")
    
    print("\nData preparation complete.")

if __name__ == "__main__":
    prepare_data() 