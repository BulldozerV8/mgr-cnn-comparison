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
from sklearn.model_selection import train_test_split, StratifiedKFold

import torchvision.transforms.functional as TF

import albumentations as A
from albumentations.pytorch import ToTensorV2

import time

import csv
import os

# --- KLASY POMOCNICZE (NA ZEWNĄTRZ DLA MULTIPROCESSINGU) ---

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


class CSVLogger:
    def __init__(self, filename="experiment_results2.csv"):
        self.filename = filename
        self.headers = [
            "type", "test_name", "dataset_tag", "model_name", "pretrained",
            "batch_size", "learning_rate", "label_smoothing", "head_depth",
            "aug_mode", "weather_prob", "aspect_ratio_mode", "num_samples",
            "use_cv", "fold", "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "test_acc", "test_prec", "test_rec", "test_f1",
            "vram_mb", "params_count", "model_size_mb", "inference_ms",
            "train_time_min", "total_fold_time_min"
        ]

        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, config, metrics, row_type="EPOCH"):
        row = [
            row_type,
            config.get("test_name", "-"), config.get("dataset_tag", "-"), config["model_name"], config["pretrained"],
            config["batch_size"], config["learning_rate"], config["label_smoothing"], config.get("head_depth", 1),
            config["aug_mode"], config.get("weather_prob", 0), config.get("aspect_ratio_mode", "Resize"),
            config["num_samples_per_epoch"], config["use_cv"], metrics.get("fold"), metrics.get("epoch", "-"),
            f"{metrics.get('train_loss', 0):.4f}", f"{metrics.get('train_acc', 0):.4f}",
            f"{metrics.get('val_loss', 0):.4f}", f"{metrics.get('val_acc', 0):.4f}",
            f"{metrics.get('test_acc', 0):.4f}", f"{metrics.get('test_prec', 0):.4f}",
            f"{metrics.get('test_rec', 0):.4f}", f"{metrics.get('test_f1', 0):.4f}",
            f"{metrics.get('vram_mb', 0):.2f}", metrics.get("params_count", 0),
            f"{metrics.get('model_size_mb', 0):.2f}", f"{metrics.get('inference_ms', 0):.4f}",
            f"{metrics.get('train_time_min', 0):.2f}", f"{metrics.get('total_fold_time_min', 0):.2f}"
        ]
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

class Predictor:
    def __init__(self, model_path, config, class_names):
        self.device = config["device"]
        self.class_names = class_names
        self.transform = TransformFactory.get_test_transforms(config)

        # Inicjalizacja modelu
        self.model = CNN(num_classes=len(class_names), model_name=config["model_name"])
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            # Zamiana na procenty (Softmax)
            probabilities = F.softmax(outputs, dim=1)[0]

        # Pobranie TOP-3
        top3_prob, top3_idx = torch.topk(probabilities, 3)

        print(f"\nWyniki analizy dla: {os.path.basename(image_path)}")
        for i in range(3):
            label = self.class_names[top3_idx[i]]
            prob = top3_prob[i].item() * 100
            print(f"{i + 1}. {label}: {prob:.2f}%")

# DODANA FUNKCJA DO PODGLĄDU ZDJĘĆ
def show_augmented_samples(loader, class_names, num_images, config): # Dodany config
    """Wyświetla przykładowe zdjęcia z loadera (po augmentacji i normalizacji) - 2 wiersze po 10"""
    images, labels = next(iter(loader))

    rows = 2
    cols = 10
    plt.figure(figsize=(22, 6))

    for i in range(min(num_images, len(images))):
        img = images[i].permute(1, 2, 0).numpy()

        # ZMIANA: Pobieranie wartości z configu zamiast sztywnego 0.485...
        mean = np.array(config["norm_mean"])
        std = np.array(config["norm_std"])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(class_names[labels[i]], fontsize=8)
        plt.axis('off')

    plt.suptitle(f"Podgląd augmentacji ({num_images} losowych próbek ze zbioru treningowego)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


class WeatherTransformWrapper:
    def __init__(self, config): # Teraz przyjmuje cały config
        self.alb_transform = A.Compose([
            # ZMIANA: scale pobierane z config
            A.RandomResizedCrop(size=(config["img_size"], config["img_size"]), scale=config["aug_scale"]),
            A.OneOf([
                A.RandomRain(p=1.0),
                A.RandomSnow(p=1.0),
                A.RandomFog(p=1.0),
            ], p=config["weather_prob"]),
            A.HorizontalFlip(p=0.5),
            # ZMIANA: mean i std z config
            A.Normalize(mean=config["norm_mean"], std=config["norm_std"]),
            ToTensorV2(),
        ])

    def __call__(self, image):
        img_np = np.array(image)
        # Albumentations zwraca słownik, wyciągamy z niego 'image'
        augmented = self.alb_transform(image=img_np)
        return augmented['image']


# --- MODUŁY KLASOWE ---

class TransformFactory:
    @staticmethod
    def get_train_transforms(config): # Przyjmuje config
        if config["aug_mode"] == "standard":
            return transforms.Compose([
                transforms.RandomResizedCrop(config["img_size"], scale=config["aug_scale"]), # to albo SquarePad + Resize
                # SquarePad(),
                # transforms.Resize((224, 224)),
                # transforms.RandomRotation(15),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # NOWE: Lekka zmiana jasności i kontrastu
                # transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),  # Automatyczny wybór mocnych transformacji (kolor, kontrast, ostrość)
                transforms.RandomHorizontalFlip(),  # Odbicie lustrzane
                transforms.ToTensor(),
                transforms.Normalize(mean=config["norm_mean"], std=config["norm_std"]),
                transforms.RandomErasing(p=config["erasing_p"], scale=(0.02, 0.2))
                # Losowe wycinanie dziur - kluczowe przy 112 powtórzeń tego samego zdjęcia
            ])
        elif config["aug_mode"] == "weather":
            return WeatherTransformWrapper(config)

    @staticmethod
    def get_test_transforms(config):  # Przyjmuje config
        return transforms.Compose([
            SquarePad(),
            transforms.Resize((config["img_size"], config["img_size"])),
            transforms.ToTensor(),
            # ZMIANA: mean i std z config
            transforms.Normalize(mean=config["norm_mean"], std=config["norm_std"])
        ])


class AircraftDataModule:
    def __init__(self, config):
        self.config = config

    def get_fold_loaders(self, train_idx, val_idx, test_idx):
        train_transforms = TransformFactory.get_train_transforms(self.config)
        test_transforms = TransformFactory.get_test_transforms(self.config)

        dataset_train_base = datasets.ImageFolder(self.config["data_path"], transform=train_transforms)
        dataset_val_base = datasets.ImageFolder(self.config["data_path"], transform=test_transforms)
        dataset_test_base = datasets.ImageFolder(self.config["data_path"], transform=test_transforms)

        train_subset = Subset(dataset_train_base, train_idx)
        val_subset = Subset(dataset_val_base, val_idx)
        test_subset = Subset(dataset_test_base, test_idx)

        # 1. Pobieramy etykiety tylko dla zdjęć z subsetu treningowego
        targets = np.array(dataset_train_base.targets)
        train_labels = targets[train_idx]

        # 2. Liczymy ile zdjęć ma każda klasa w treningu
        class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(targets)])

        # 3. Waga zdjęcia = 1 / liczność klasy (im rzadsza klasa, tym większa szansa na wylosowanie)
        weight = 1. / class_sample_count
        samples_weight = torch.from_numpy(np.array([weight[t] for t in train_labels]))

        # 4. 30 000 kroków na epokę (każda klasa dostanie średnio tyle samo szans)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=samples_weight,
            num_samples=self.config["num_samples_per_epoch"],
            replacement=True
        )

        # DataLoader z wykorzystaniem Samplera i procesora do augmentacji w tle
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config["batch_size"], # ZMIANA
            sampler=sampler,
            num_workers=self.config["num_workers"],
            pin_memory=True
        )
        # ZMIANA: self.batch_size -> self.config["batch_size"]
        val_loader = DataLoader(val_subset, batch_size=self.config["batch_size"], shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=self.config["batch_size"], shuffle=False)

        return train_loader, val_loader, test_loader



class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=3, min_delta=0.005):
        """
        Args:
            patience (int): Ile epok czekać po ostatniej poprawie.
            min_delta (float): Minimalna zmiana traktowana jako faktyczna poprawa.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        # Sprawdzamy, czy obecny błąd jest ZNACZĄCO mniejszy od najlepszego
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0 # Reset licznika, bo nastąpiła poprawa
        else:
            self.counter += 1
            print(f"    [EarlyStopping] Brak znaczącej poprawy (min_delta={self.min_delta}). Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
class TrainingEngine:
    def __init__(self, model, device, criterion, optimizer, scheduler):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_one_epoch(self, loader, num_classes, epoch, num_epochs, fold):
        self.model.train()
        running_loss = 0
        acc_train = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)

        for data, targets in tqdm(loader, desc=f"Fold {fold} Ep {epoch}/{num_epochs}"):
            data, targets = data.to(self.device), targets.to(self.device)
            scores = self.model(data)
            loss = self.criterion(scores, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(scores.data, 1)
            acc_train.update(predicted, targets)

        return running_loss / len(loader), acc_train.compute().item()

    def validate(self, loader, num_classes):
        # Walidacja
        self.model.eval()
        val_running_loss = 0
        acc_val = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                acc_val.update(predicted, labels)

        return val_running_loss / len(loader), acc_val.compute().item()


class Visualizer:
    @staticmethod
    def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, fold):
        # --- WIZUALIZACJA ---
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Fold {fold}: Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title(f'Fold {fold}: Training vs Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm_percent, class_names, fold):
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm_percent.numpy(), xticklabels=class_names, yticklabels=class_names, cmap="viridis", annot=False)
        plt.title(f"Fold {fold} Confusion Matrix (%)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()


class Evaluator:
    @staticmethod
    def evaluate_and_plot(model, test_loader, criterion, device, num_classes, class_names, fold):
        # Testowanie
        acc_test = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        prec_test = Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
        rec_test = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
        f1_test = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
        confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

        total_loss, num_batches = 0, 0
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
        cm = confusion_matrix.compute().cpu().float()
        cm_percent = cm / cm.sum(dim=1, keepdim=True)

        print(f"\nFold {fold} Test Loss: {total_loss / num_batches:.6f}")
        print(f"Fold {fold} Test Acc: {test_acc:.4f} | Prec: {prec_test.compute().item():.4f} | Rec: {rec_test.compute().item():.4f} | F1: {f1_test.compute().item():.4f}")

        Visualizer.plot_confusion_matrix(cm_percent, class_names, fold)

        errors = []
        for true_idx in range(num_classes):
            for pred_idx in range(num_classes):
                if true_idx != pred_idx:
                    count = int(cm[true_idx, pred_idx].item())
                    if count > 0: errors.append((count, true_idx, pred_idx))
        errors.sort(reverse=True, key=lambda x: x[0])
        print(f"\nTOP 10 najczęstszych pomyłek dla Fold {fold}:\n")
        for i in range(min(10, len(errors))):
            count, t, p = errors[i]
            print(f"{i + 1}. {class_names[t]} → {class_names[p]} : {count} razy")

        results = {
            'acc': acc_test.compute().item(),
            'prec': prec_test.compute().item(),
            'rec': rec_test.compute().item(),
            'f1': f1_test.compute().item()
        }

        # Wyświetlanie w konsoli (zostaje dla Ciebie)
        print(f"\nFold {fold} Test Acc: {results['acc']:.4f} | F1: {results['f1']:.4f}")

        return results


class ExperimentRunner:
    def __init__(self, config):
        self.config = config  # Zapisujemy cały słownik

    def run_cross_validation(self):
        import time
        if self.config["use_seed"]:
            set_seed(self.config["seed"])
            print(f"Ziarno ustawione na: {self.config['seed']}")
        else:
            print("Ziarno wyłączone - pełna losowość systemu.")

        # Inicjalizacja Loggera przed pętlą
        logger = CSVLogger()
        start_total = time.time()

        temp_dataset = datasets.ImageFolder(self.config["data_path"])
        targets = np.array(temp_dataset.targets)
        indices = np.arange(len(targets))
        class_names = temp_dataset.classes
        num_classes = len(class_names)

        # 1. Stały zbiór testowy - używamy test_size z configu
        cv_indices, test_idx = train_test_split(
            indices,
            test_size=self.config["test_size"],
            shuffle=True,
            stratify=targets,
            random_state=self.config["seed"] if self.config["use_seed"] else None
        )
        cv_targets = targets[cv_indices]

        # 2. Wybór trybu (CV vs Simple Split)
        if self.config["use_cv"]:
            skf = StratifiedKFold(n_splits=self.config["num_folds"], shuffle=True,
                                  random_state=self.config["seed"] if self.config["use_seed"] else None)
            folds_to_run = list(skf.split(cv_indices, cv_targets))
            print(f"TRYB: Walidacja Krzyżowa ({self.config['num_folds']} foldów)")
        else:
            # Używamy val_size z configu
            t_idx, v_idx = train_test_split(
                np.arange(len(cv_indices)),
                test_size=self.config["val_size"],
                stratify=cv_targets,
                random_state=self.config["seed"] if self.config["use_seed"] else None
            )
            folds_to_run = [(t_idx, v_idx)]
            print(f"TRYB: Szybki (Tylko 1 fold - CV wyłączone)")

        cv_scores = []

        # 3. PĘTLA PO FOLDACH
        for fold, (train_fold_idx, val_fold_idx) in enumerate(folds_to_run):
            fold_num = fold + 1
            print(f"\n{'=' * 20} FOLD {fold_num}/{len(folds_to_run)} {'=' * 20}")

            start_fold_total = time.time()  # STOPER 1: CAŁY EKSPERYMENT (od ładowania danych)

            data_module = AircraftDataModule(self.config)

            train_loader, val_loader, test_loader = data_module.get_fold_loaders(
                cv_indices[train_fold_idx], cv_indices[val_fold_idx], test_idx
            )

            # --- PRZYWRÓCONY FRAGMENT Z INFORMACJAMI O DANYCH ---
            if fold == 0:
                train_subset_len = len(train_fold_idx)
                val_subset_len = len(val_fold_idx)

                print(f"Wykryto klas: {num_classes}")
                print(f"Zbiór fizyczny (train): {train_subset_len} zdjęć")
                print(f"Zbiór efektywny (per epoch): {self.config['num_samples_per_epoch']} zdjęć")
                print(f"Cel na klasę (per epoch): ~{self.config['num_samples_per_epoch'] // num_classes} zdjęć")
                print(f"Walidacja: {val_subset_len} zdjęć (bez augmentacji i oversamplingu)")

                show_augmented_samples(train_loader, class_names, 20, self.config)

            # Inicjalizacja sieci i silnika (DODANO head_depth do inicjalizacji)
            model = CNN(
                num_classes=num_classes,
                model_name=self.config["model_name"],
                pretrained=self.config["pretrained"],
                head_depth=self.config.get("head_depth", 1)  # <--- DODANO
            ).to(self.config["device"])

            criterion = nn.CrossEntropyLoss(label_smoothing=self.config["label_smoothing"])

            # Pomiar złożoności (liczba parametrów)
            params_count = sum(p.numel() for p in model.parameters())

            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',
                factor=self.config["scheduler_factor"],
                patience=self.config["scheduler_patience"]
            )

            early_stopping = EarlyStopping(
                patience=self.config["early_stopping_patience"],
                min_delta=self.config["early_stopping_min_delta"]
            )

            engine = TrainingEngine(model, self.config["device"], criterion, optimizer, scheduler)

            train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

            start_training = time.time()  # STOPER 2: CZYSTY TRENING (same epoki)

            # Pętla epok
            for epoch in range(self.config["num_epochs"]):
                t_loss, t_acc = engine.train_one_epoch(train_loader, num_classes, epoch + 1, self.config["num_epochs"],
                                                       fold_num)
                v_loss, v_acc = engine.validate(val_loader, num_classes)

                engine.scheduler.step(v_loss)

                train_losses.append(t_loss)
                val_losses.append(v_loss)
                train_accuracies.append(t_acc)
                val_accuracies.append(v_acc)

                # Pomiar VRAM (Hardware Profiling)
                vram_mb = torch.cuda.max_memory_allocated(self.config["device"]) / (1024 ** 2) if "cuda" in self.config[
                    "device"] else 0

                # LOGOWANIE DO CSV (Zapis po każdej epoce)
                logger.log(self.config, {
                    "fold": fold_num, "epoch": epoch + 1,
                    "train_loss": t_loss, "train_acc": t_acc,
                    "val_loss": v_loss, "val_acc": v_acc,
                    "vram_mb": vram_mb, "params_count": params_count,
                    "train_time_min": (time.time() - start_training) / 60  # <--- DODANO
                })

                print(
                    f"Fold {fold_num} Ep {epoch + 1}: "
                    f"Train Loss={t_loss:.4f}, Train Acc={t_acc:.4f} | "
                    f"Val Loss={v_loss:.4f}, Val Acc={v_acc:.4f} | "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}"
                )

                if self.config.get("use_early_stopping", True):
                    early_stopping(v_loss)
                    if early_stopping.early_stop:
                        print(f"    [EarlyStopping] Zatrzymano trening foldu {fold_num} na epoce {epoch + 1}.")
                        break
                pass

            training_time_min = (time.time() - start_training) / 60  # <--- KONIEC STOPER 2

            # Rysowanie, Testowanie i Zapis z osobnych modułów
            Visualizer.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, fold_num)
            test_metrics = Evaluator.evaluate_and_plot(model, test_loader, criterion, self.config["device"],
                                                       num_classes, class_names, fold_num)

            # --- DODATKOWE MIARY PRZED LOGOWANIEM SUMMARY ---

            # 1. Rozmiar modelu na dysku
            model_path = f"model_weights_fold_{fold_num}.pth"
            torch.save(model.state_dict(), model_path)
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

            # 2. Pomiar czasu inferencji (średnia z 10 zdjęć)
            model.eval()
            dummy_input = torch.randn(1, 3, self.config["img_size"], self.config["img_size"]).to(self.config["device"])
            start_inf = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            inference_ms = ((time.time() - start_inf) / 10) * 1000  # Wynik w milisekundach

            total_fold_time_min = (time.time() - start_fold_total) / 60  # <--- KONIEC STOPER 1

            # 3. Zapis do Loggera (Summary)
            logger.log(self.config, {
                "fold": fold_num,
                "test_acc": test_metrics['acc'],
                "test_prec": test_metrics['prec'],
                "test_rec": test_metrics['rec'],
                "test_f1": test_metrics['f1'],
                "vram_mb": torch.cuda.max_memory_allocated(self.config["device"]) / (
                        1024 ** 2) if torch.cuda.is_available() else 0,
                "params_count": params_count,
                "model_size_mb": model_size_mb,
                "inference_ms": inference_ms,
                "train_time_min": training_time_min,  # <--- DODANO
                "total_fold_time_min": total_fold_time_min  # <--- DODANO
            }, row_type="SUMMARY")

            cv_scores.append(test_metrics['acc'])
            torch.save(model.state_dict(), f"model_weights_fold_{fold_num}.pth")

        end_total = time.time()
        total_duration = (end_total - start_total) / 60  # Całkowity czas w minutach

        print(f"\n{'#' * 40}")
        print(f"KONIEC EKSPERYMENTU")
        print(f"Całkowity czas uczenia: {total_duration:.2f} minut ({total_duration / 60:.2f} h)")
        print(f"ŚREDNIA SKUTECZNOŚĆ: {np.mean(cv_scores) * 100:.2f}% ± {np.std(cv_scores) * 100:.2f}%")
        print(f"{'#' * 40}")

        # Zapis nazw klas do JSON
        import json
        with open("class_names.json", "w") as f:
            json.dump(class_names, f)


class CNN(nn.Module):
    def __init__(self, num_classes, model_name="efficientnet_b0", pretrained=True, head_depth=1, head_width=512):
        super(CNN, self).__init__()

        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
            in_features = self.model.classifier[1].in_features

            # Budujemy głowicę dynamicznie
            layers = []
            curr_in = in_features
            # Jeśli depth > 1, dodajemy warstwy pośrednie
            for _ in range(head_depth - 1):
                layers.append(nn.Linear(curr_in, 512))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))
                curr_in = 512

            # Ostatnia warstwa klasyfikująca
            layers.append(nn.Linear(curr_in, num_classes))
            self.model.classifier[1] = nn.Sequential(*layers)

        elif model_name == "my_custom_cnn":
            # AUTORSKI MODEL (Krok 4)
            pass

        elif model_name == "resnet18":
            # KROK 5 (Inny gotowiec) (aktualnie losowy resnet, potem do zmiany)
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Nieznana architektura modelu: {model_name}")

    def forward(self, x):
        return self.model(x)

# --- FUNKCJA MAIN ---

def main():
    # --- KONFIGURACJA EKSPERYMENTU (CENTRALNA) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Wykryte urządzenie: {device}")

    config = {
        "test_name": "EffNet_Baseline",  # Nazwa testu do Excela
        "dataset_tag": "Satelitarne_10",  # Etykieta zbioru danych
        "aspect_ratio_mode": "SquarePad",  # "SquarePad" lub "Resize"
        "head_depth": 1,  # 1 = standard, 2 lub 3 = dodatkowe warstwy
        "head_width": 512,  # Szerokość dodatkowych warstw

        # --- DANE I PODZIAŁY ---
        "data_path": "airplane10/",
        "aug_mode": "standard", # standard lub weather
        "weather_prob": 0.7,
        "test_size": 0.15,  # Stały zbiór testowy na sam koniec (hold-out)
        "val_size": 0.20,  # Rozmiar walidacji (tylko gdy use_cv = False)
        "use_cv": False,  # True = 5-fold CV, False = pojedynczy fold (szybki test)
        "num_folds": 5,  # Liczba foldów dla CV
        "use_seed": False,
        "seed": 50,  # Ziarno losowości dla powtarzalności wyników

        # --- PARAMETRY TECHNICZNE ---
        "img_size": 224,  # Rozdzielczość wejściowa dla modelu
        "batch_size": 64,  # Wielkość paczki danych
        "num_workers": 4,  # Liczba procesów CPU do ładowania zdjęć
        "model_name": "efficientnet_b0",
        "pretrained": True,  # Czy model ma ładować wagi ImageNet (Transfer Learning)

        # --- PARAMETRY UCZENIA (HIPERPARAMETRY) ---
        "num_epochs": 5,
        "num_samples_per_epoch": 30000,  # Ile zdjęć model widzi w jednej epoce (Oversampling)
        "learning_rate": 0.0001,
        "weight_decay": 1e-4,  # Regularyzacja L2
        "label_smoothing": 0.1,  # Wygładzanie etykiet (zapobiega przeuczeniu)

        # --- EARLY STOPPING I SCHEDULER ---
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.005,
        "use_early_stopping": False,
        "scheduler_patience": 1,
        "scheduler_factor": 0.5,

        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
        "aug_scale": (0.8, 1.0),
        "erasing_p": 0.5,
        "device": device
    }

    # --- URUCHOMIENIE POTOKU BADAWCZEGO ---
    runner = ExperimentRunner(config)
    runner.run_cross_validation()



if __name__ == '__main__':
    main()