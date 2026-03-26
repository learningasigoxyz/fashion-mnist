"""
Benchmark GLM (Logistic Regression), XGBoost, and CNN on Fashion-MNIST.

Usage:
    python benchmark/benchmark_glm_xgb_cnn.py
"""

import sys
import time
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import mnist_reader

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fashion")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_data():
    X_train, y_train = mnist_reader.load_mnist(DATA_DIR, kind="train")
    X_test, y_test = mnist_reader.load_mnist(DATA_DIR, kind="t10k")
    return X_train, y_train, X_test, y_test


def benchmark_glm(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.astype(np.float32))
    X_te = scaler.transform(X_test.astype(np.float32))

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, n_jobs=-1)

    t0 = time.time()
    clf.fit(X_tr, y_train)
    train_time = time.time() - t0

    acc = clf.score(X_te, y_test)
    return acc, train_time


def benchmark_xgboost(X_train, y_train, X_test, y_test):
    import xgboost as xgb

    X_tr = X_train.astype(np.float32) / 255.0
    X_te = X_test.astype(np.float32) / 255.0

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )

    t0 = time.time()
    clf.fit(X_tr, y_train)
    train_time = time.time() - t0

    acc = clf.score(X_te, y_test)
    return acc, train_time


def benchmark_cnn(X_train, y_train, X_test, y_test):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reshape to (N, 1, 28, 28) and normalise
    X_tr = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
    y_te = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_tr, y_tr)
    test_ds = TensorDataset(X_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),                             # 14x14
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),                             # 7x7
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 7 * 7, 256), nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    epochs = 15
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()
        avg_loss = running_loss / len(train_ds)
        print(f"  Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}", flush=True)
    train_time = time.time() - t0

    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
    acc = correct / len(test_ds)
    return acc, train_time


def print_result(name, acc, train_time):
    print(f"\n{'='*50}")
    print(f"Model        : {name}")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Train Time   : {train_time:.1f}s")
    print(f"{'='*50}")


def main():
    print("Loading Fashion-MNIST data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    results = {}

    print("\n--- GLM (Logistic Regression) ---")
    acc, t = benchmark_glm(X_train, y_train, X_test, y_test)
    results["GLM"] = (acc, t)
    print_result("GLM (Logistic Regression)", acc, t)

    print("\n--- XGBoost ---")
    acc, t = benchmark_xgboost(X_train, y_train, X_test, y_test)
    results["XGBoost"] = (acc, t)
    print_result("XGBoost", acc, t)

    print("\n--- CNN (PyTorch) ---")
    acc, t = benchmark_cnn(X_train, y_train, X_test, y_test)
    results["CNN"] = (acc, t)
    print_result("CNN (PyTorch)", acc, t)

    print("\n\n===== SUMMARY =====")
    print(f"{'Model':<25} {'Accuracy':>10} {'Train Time':>12}")
    print("-" * 50)
    for name, (acc, t) in results.items():
        print(f"{name:<25} {acc*100:>9.2f}% {t:>10.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
