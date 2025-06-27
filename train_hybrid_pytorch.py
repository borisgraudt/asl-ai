import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Синтетический датасет XOR (100% достижимо) ---
def make_xor(n=200):
    X = np.random.randint(0, 2, (n, 2))
    y = (X[:, 0] ^ X[:, 1])
    return X.astype(np.float32), y.astype(np.int64)

# --- Классическая модель ---
class ClassicalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Обучение и тест ---
def train_and_eval(model, X_train, y_train, X_test, y_test, epochs=30, lr=0.1, batch_size=32):
    device = torch.device('cpu')
    model = model.to(device)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_accs, test_accs = [], []
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # Accuracy
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train).argmax(1)
            test_preds = model(X_test).argmax(1)
            train_acc = accuracy_score(y_train.cpu(), train_preds.cpu())
            test_acc = accuracy_score(y_test.cpu(), test_preds.cpu())
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")
    return train_accs, test_accs

if __name__ == "__main__":
    # XOR demo
    X, y = make_xor(200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\n=== Classical ===")
    cnet = ClassicalNet()
    c_train_acc, c_test_acc = train_and_eval(cnet, X_train, y_train, X_test, y_test)
    # Plot
    plt.plot(c_test_acc, label='Classical')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.show() 