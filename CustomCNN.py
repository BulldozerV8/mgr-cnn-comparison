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

#set_seed(50)

data_path = "Dataset-small-perfect/"

#Wybór urzadzenia, jesli dostepne to gpu, jesli nie, to cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 16
num_epochs = 15

#TRAIN transforms: resize + lekkie augmentacje + normalizacja
train_transforms = transforms.Compose([
    #transforms.RandomCrop((100,100)),
    SquarePad(),
    transforms.Resize((224, 224)),
    #transforms.RandomResizedCrop(100, scale=(0.92, 1.0)),
    transforms.RandomRotation(15),           # minimalne obroty dla uodpornienia modelu na orientację
    #transforms.ColorJitter(brightness=0.12),# drobne różnice w jasności, symuluje różne warunki oświetleniowe
    transforms.RandomHorizontalFlip(),       # losowe odbicie poziome, ułatwia generalizację
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#TEST transforms: tylko resize + normalizacja (bez augmentacji)
test_transforms = transforms.Compose([
    SquarePad(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#Ładowanie baz
dataset_train_base = datasets.ImageFolder(data_path, transform=train_transforms)
dataset_val_base = datasets.ImageFolder(data_path, transform=test_transforms)
dataset_test_base = datasets.ImageFolder(data_path, transform=test_transforms)

# Dynamiczne zczytanie klas
class_names = dataset_train_base.classes
num_classes = len(class_names)
print(f"Wykryto klas: {num_classes}")

targets = dataset_train_base.targets

#Odcięcie 15% na zbiór testowy
train_val_idx, test_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.15,
    shuffle=True,
    stratify=targets,
    random_state=42
)

train_val_targets = [targets[i] for i in train_val_idx]

#Podział reszty (85%) na Train i Val (80/20 z tej części)
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


def imshow(img):
    npimg = img.numpy()
    #Cofnięcie normalizacji do podglądu
    npimg = npimg / 2 + 0.5
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#Pokazanie losowych danych treningowych
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(labels)
imshow(torchvision.utils.make_grid(images))

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # stabilizacja rozkładu aktywacji
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)


        #self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.bn2 = nn.BatchNorm2d(64)

        # Policzenie rozmiaru
        x = torch.randn(1, in_channels, 224, 224)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))
        fc_input_size = x.numel() // x.shape[0]

        self.fc1 = nn.Sequential(
            nn.Dropout(0.3),                #Wyłączanie losowych neuronów
            nn.Linear(in_features=fc_input_size, out_features=num_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


model = CNN(in_channels=3, num_classes=num_classes).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# .to(device) jest kluczowe, inaczej wywali błąd przy metrykach!
acc_train = Accuracy(task="multiclass", num_classes=num_classes).to(device)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    #print("Epoch {}/{}".format(epoch + 1, num_epochs))
    model.train()
    running_loss = 0
    acc_train.reset()

    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data) #Obliczenie predykcji modelu
        loss = criterion(scores, targets) #Obliczenie loss
        optimizer.zero_grad() #Zerowanie gradientu z poprzedniego kroku
        loss.backward() #Wsteczna propagacja bledu, obliczenie gradientu
        optimizer.step() #Aktualizacja wag modelu wedlug gradientow i optymalizatora

        running_loss += loss.item()

        _, predicted = torch.max(scores.data, 1)
        acc_train.update(predicted, targets)

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

    print(f"Epoch {epoch + 1}: Val Loss={avg_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}")
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

acc_test = Accuracy(task= "multiclass", num_classes=num_classes) #Accuracy – jaki procent wszystkich próbek został sklasyfikowany poprawnie
prec_test = Precision(task= "multiclass", num_classes=num_classes, average="macro") #Precision (macro) – średnio dla wszystkich klas: jak często model ma rację, gdy przewidzi daną klasę
rec_test = Recall(task= "multiclass", num_classes=num_classes, average="macro") #Recall (macro) – średnio dla wszystkich klas: jaki procent rzeczywistych próbek danej klasy został wykryty
f1_test = F1Score(task= "multiclass", num_classes=num_classes, average="macro") #F1-score (macro) – kompromis między precision i recall, wrażliwy na błędy w klasach rzadkich
confusion_matrix = ConfusionMatrix(task= "multiclass", num_classes=num_classes) #Confusion Matrix – macierz pokazująca, które klasy są poprawnie rozpoznawane, a które są ze sobą mylone
total_loss = 0 #Suma wartości funkcji straty na zbiorze testowym (do obliczenia średniego lossu)
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

test_acc = acc_test.compute()
test_prec = prec_test.compute()
test_rec = rec_test.compute()
test_f1 = f1_test.compute()
confusion_matrix = confusion_matrix.compute()
avg_loss = total_loss / num_batches

print("Test Loss: {:.6f}".format(avg_loss))
print(f"Test accuracy: {test_acc}")
print(f"Test precision: {test_prec}")
print(f"Test recall: {test_rec}")
print(f"Test F1: {test_f1}")
print(f"Confusion matrix: {confusion_matrix}")


#Nazwy klas
class_names = dataset_test_base.classes
num_classes = len(class_names)

#Konwersja confusion matrix na CUDA → CPU i na float
cm = confusion_matrix.cpu().float()

#Confusion matrix w %
cm_percent = cm / cm.sum(dim=1, keepdim=True)

#Heatmapa confusion matrix w %
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

#TOP-10 najczęstszych pomyłek
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
    print(f"{i+1}. {class_names[t]} → {class_names[p]} : {count} razy")