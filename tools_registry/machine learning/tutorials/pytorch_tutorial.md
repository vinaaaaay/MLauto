Summary: Comprehensive PyTorch deep learning guide covering custom Datasets, DataLoaders, model architectures (nn.Module), classification/regression loss functions, optimization, training loops, and validation/evaluation procedures.

# PyTorch Training Tutorial

A practical reference for training models with PyTorch. Covers the full loop: data → model → loss → optimizer → train → evaluate.

---

## Core Concepts (Always True)

Every PyTorch training job follows this skeleton:

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Define Dataset
# 2. Define Model (nn.Module)
# 3. Pick Loss function
# 4. Pick Optimizer
# 5. Training loop: forward → loss → backward → step
# 6. Evaluation loop (no gradients)
```

**Tensor dtypes matter:**
- Features: `torch.float32`
- Regression targets: `torch.float32`
- Classification targets: `torch.long` (integer class indices)

**Always move model and data to the same device:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X, y = X.to(device), y.to(device)
```

---

## 1. Regression

**Problem:** Predict a continuous value (e.g., house price, temperature).

**Loss:** `nn.MSELoss()` (mean squared error). Use `nn.L1Loss()` if outliers are a concern.

**Output layer:** Linear, no activation.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- Dataset ---
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        # X: numpy array shape (N, num_features)
        # y: numpy array shape (N,)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Model ---
class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # single output, no activation
        )

    def forward(self, x):
        return self.net(x)

# --- Setup ---
# Replace with real data
X_train = np.random.randn(1000, 10).astype(np.float32)
y_train = np.random.randn(1000).astype(np.float32)

dataset = RegressionDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegressionNet(input_dim=10).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training loop ---
for epoch in range(50):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        preds = model(X_batch)          # forward
        loss = loss_fn(preds, y_batch)  # compute loss

        optimizer.zero_grad()           # clear old gradients
        loss.backward()                 # backprop
        optimizer.step()                # update weights

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")
```

**Evaluation:**
```python
model.eval()
with torch.no_grad():
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    predictions = model(X_test_t).squeeze().cpu().numpy()
```

---

## 2. Binary Classification

**Problem:** Predict one of two classes (e.g., spam/not-spam).

**Loss:** `nn.BCEWithLogitsLoss()` — takes raw logits (no sigmoid in model).

**Output layer:** Linear with 1 output, **no sigmoid** (the loss handles it).

**Targets:** `torch.float32`, values 0.0 or 1.0.

```python
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)    # raw logit, NO sigmoid here
        )

    def forward(self, x):
        return self.net(x)

# --- Dataset ---
class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --- Setup ---
model = BinaryClassifier(input_dim=10).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training loop (same skeleton as regression) ---
for epoch in range(50):
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Getting predictions:**
```python
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    probs = torch.sigmoid(logits)       # probability of class 1
    preds = (probs >= 0.5).long()       # binary class labels
```

---

## 3. Multi-Class Classification

**Problem:** Predict one of N classes (e.g., digit recognition, 10 classes).

**Loss:** `nn.CrossEntropyLoss()` — takes raw logits of shape `(batch, num_classes)`.

**Output layer:** Linear with `num_classes` outputs, **no softmax** (the loss handles it).

**Targets:** `torch.long`, integer class indices (0 to num_classes-1). Shape `(N,)`, not one-hot.

```python
class MultiClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # raw logits, NO softmax
        )

    def forward(self, x):
        return self.net(x)

# --- Dataset ---
class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # long, NOT float, shape (N,)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --- Setup ---
num_classes = 5
model = MultiClassifier(input_dim=20, num_classes=num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training loop ---
for epoch in range(50):
    model.train()
    correct, total = 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)             # shape: (batch, num_classes)
        loss = loss_fn(logits, y_batch)     # y_batch shape: (batch,)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    print(f"Epoch {epoch+1} | Acc: {correct/total:.4f}")
```

**Getting predictions:**
```python
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    probs = torch.softmax(logits, dim=1)  # class probabilities
    preds = logits.argmax(dim=1)          # predicted class indices
```

---

## 4. Image Classification (CNN)

**Problem:** Classify images (e.g., CIFAR-10, MNIST).

**Input shape:** `(batch, channels, height, width)` — channels first.

**Loss:** `nn.CrossEntropyLoss()` with long targets (same as multi-class).

```python
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (B, 32, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B, 32, H/2, W/2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B, 64, H/4, W/4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),  # adjust 8*8 to match your image size after pooling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- Dataset using torchvision ---
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # per-channel mean, std
])

train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)

# --- Training ---
model = ConvNet(num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 5. Using a Pretrained Model (Transfer Learning)

Faster and usually more accurate than training from scratch on small datasets.

```python
from torchvision import models

# Load pretrained ResNet18
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze all layers (only train the new head)
for param in model.parameters():
    param.requires_grad = False

# Replace the final classification head
num_features = model.fc.in_features      # 512 for ResNet18
model.fc = nn.Linear(num_features, 10)   # 10 = your number of classes

model = model.to(device)

# Only pass parameters that require gradients to optimizer
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

# Training loop is identical to multi-class classification above
```

**To fine-tune all layers instead of just the head:**
```python
# Unfreeze all layers after initial head training
for param in model.parameters():
    param.requires_grad = True

# Use a smaller LR for fine-tuning to avoid destroying pretrained weights
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

---

## 6. Sequence Model (LSTM for Time Series / NLP)

**Problem:** Predict from sequential data (e.g., time series forecasting, text classification).

**Input shape:** `(batch, seq_len, input_size)` when `batch_first=True`.

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,    # input shape: (batch, seq, features)
            dropout=0.2          # only applies if num_layers > 1
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # Use last timestep's output for classification/regression
        last = out[:, -1, :]          # shape: (batch, hidden_size)
        return self.fc(last)

# --- Example: time series classification ---
# X shape: (N, seq_len=50, features=1)
# y shape: (N,) with class indices

model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, num_classes=3).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

---

## 7. Learning Rate Scheduling

Decay the learning rate during training for better convergence.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Option A: StepLR — multiply LR by gamma every step_size epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Option B: ReduceLROnPlateau — reduce when a metric stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

for epoch in range(100):
    train(...)                          # your training loop

    # StepLR: call unconditionally each epoch
    scheduler.step()

    # ReduceLROnPlateau: pass the metric you are monitoring
    # scheduler.step(val_loss)
```

---

## 8. Saving and Loading Models

```python
# Save only weights (recommended)
torch.save(model.state_dict(), "model.pth")

# Load weights — must instantiate model class first
model = MyModel(...).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Save full checkpoint (weights + optimizer + epoch)
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "loss": loss,
}, "checkpoint.pth")

# Resume from checkpoint
checkpoint = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
start_epoch = checkpoint["epoch"]
```

---

## 9. Validation Loop

Always evaluate on a held-out set to detect overfitting. Run after each training epoch.

```python
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():                       # no gradient tracking
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            total_loss += loss.item()

            # For classification only:
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total              # omit for regression
    return avg_loss, accuracy

# Usage in training loop:
for epoch in range(100):
    train_one_epoch(model, train_loader, ...)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
    print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
```

---

## Common Mistakes & Fixes

| Mistake | Fix |
|---|---|
| `RuntimeError: Expected target size (N, C) got (N,)` | Targets for `CrossEntropyLoss` must be shape `(N,)` of type `long`, not one-hot |
| `RuntimeError: expected scalar type Long but found Float` | Classification targets need `dtype=torch.long` |
| Model not learning | Check `optimizer.zero_grad()` is inside the loop, not outside |
| Loss is NaN | Learning rate too high, or inputs contain NaN/Inf — check with `torch.isnan(X).any()` |
| `BCELoss` with values outside [0,1] | Use `BCEWithLogitsLoss` with raw logits instead of `BCELoss` with sigmoid output |
| Validation accuracy not tracked correctly | Wrap eval loop in `model.eval()` and `torch.no_grad()` |
| GPU out of memory | Reduce batch size, or call `torch.cuda.empty_cache()` after eval |
| Slow training on CPU | Add `num_workers=4` to DataLoader (use 0 on Windows if errors occur) |
