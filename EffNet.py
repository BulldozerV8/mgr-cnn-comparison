import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torchvision.transforms.functional as TF


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(image, padding, 0, 'constant')


def set_seed(seed=50):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# set_seed(50)

# ZMIENIONA ŚCIEŻKA NA NOWĄ BAZĘ
data_path = "CUB/"

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 16
num_epochs = 15  # Zwiększono do 15, bo z Schedulerem model może uczyć się dłużej i dokładniej

# TRAIN transforms: Dodano ColorJitter (Regularyzacja)
train_transforms = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # NOWE: Lekka zmiana jasności i kontrastu
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_train_base = datasets.ImageFolder(data_path, transform=train_transforms)
dataset_val_base = datasets.ImageFolder(data_path, transform=test_transforms)
dataset_test_base = datasets.ImageFolder(data_path, transform=test_transforms)

class_names = dataset_train_base.classes
num_classes = len(class_names)
print(f"Wykryto klas: {num_classes}")

targets = dataset_train_base.targets

train_val_idx, test_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.15,
    shuffle=True,
    stratify=targets,
    random_state=42
)

train_val_targets = [targets[i] for i in train_val_idx]

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.2,
    shuffle=True,
    stratify=train_val_targets,
    random_state=42
)

train_subset = Subset(dataset_train_base, train_idx)
val_subset = Subset(dataset_val_base, val_idx)
test_subset = Subset(dataset_test_base, test_idx)

print(f"Trening: {len(train_subset)} zdjęć")
print(f"Walidacja: {len(val_subset)} zdjęć")
print(f"Test: {len(test_subset)} zdjęć")

train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.model = models.efficientnet_b0(weights=weights)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


model = CNN(num_classes=num_classes).to(device)
#print(model)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# NOWE: Harmonogram uczenia (Scheduler). Jeśli przez 2 epoki Val Loss nie spadnie, dzieli LR przez 2.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

acc_train = Accuracy(task="multiclass", num_classes=num_classes).to(device)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    acc_train.reset()

    for batch_index, (data, targets_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        data = data.to(device)
        targets_batch = targets_batch.to(device)

        scores = model(data)
        loss = criterion(scores, targets_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(scores.data, 1)
        acc_train.update(predicted, targets_batch)

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    epoch_train_acc = acc_train.compute().item()
    train_accuracies.append(epoch_train_acc)

    # Walidacja
    model.eval()
    val_running_loss = 0
    acc_val = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            acc_val.update(predicted, labels)

    avg_val_loss = val_running_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    epoch_val_acc = acc_val.compute().item()
    val_accuracies.append(epoch_val_acc)

    # NOWE: Aktualizacja Schedulera na podstawie błędu walidacji
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch + 1}: Val Loss={avg_val_loss:.4f}, Val Acc={epoch_val_acc:.4f} | LR={current_lr:.6f}")
    print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Train Acc={epoch_train_acc:.4f}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Testowanie
acc_test = Accuracy(task="multiclass", num_classes=num_classes).to(device)
prec_test = Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
rec_test = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
f1_test = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

total_loss = 0
num_batches = 0

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        total_loss += loss.item()
        num_batches += 1

        _, predicted = torch.max(outputs.data, 1)
        acc_test.update(predicted, labels)
        prec_test.update(predicted, labels)
        rec_test.update(predicted, labels)
        f1_test.update(predicted, labels)
        confusion_matrix.update(predicted, labels)

test_acc = acc_test.compute().item()
test_prec = prec_test.compute().item()
test_rec = rec_test.compute().item()
test_f1 = f1_test.compute().item()
confusion_matrix = confusion_matrix.compute()
avg_loss = total_loss / num_batches

print("Test Loss: {:.6f}".format(avg_loss))
print(f"Test accuracy: {test_acc}")
print(f"Test precision: {test_prec}")
print(f"Test recall: {test_rec}")
print(f"Test F1: {test_f1}")

# Nazwy klas
class_names = dataset_train_base.classes
num_classes = len(class_names)

cm = confusion_matrix.cpu().float()
cm_percent = cm / cm.sum(dim=1, keepdim=True)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm_percent.numpy(),
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="viridis",
    annot=False
)
plt.title("Confusion Matrix (%)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

errors = []
for true_idx in range(num_classes):
    for pred_idx in range(num_classes):
        if true_idx != pred_idx:
            count = int(cm[true_idx, pred_idx].item())
            if count > 0:
                errors.append((count, true_idx, pred_idx))

errors.sort(reverse=True, key=lambda x: x[0])

print("\nTOP 10 najczęstszych pomyłek:\n")
for i in range(min(10, len(errors))):
    count, t, p = errors[i]
    print(f"{i + 1}. {class_names[t]} → {class_names[p]} : {count} razy")