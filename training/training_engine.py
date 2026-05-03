import torch
from torchmetrics import Accuracy
from tqdm import tqdm


class TrainingEngine:
    """
    Klasa zarządzająca niskopoziomową logiką procesu uczenia i ewaluacji modelu.
    Implementuje nowoczesne techniki optymalizacji obliczeń, takie jak AMP oraz
    efektywne zarządzanie pamięcią GPU.
    """

    def __init__(self, model, device, criterion, optimizer, scheduler, num_classes):
        """
        Inicjalizuje silnik treningowy, konfigurując podzespoły optymalizacyjne i metryki.
        """
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Inicjalizacja GradScalera dla automatycznej mieszanej precyzji (AMP)
        self.scaler = torch.amp.GradScaler()

        # Jednorazowa inicjalizacja metryk na GPU w celu uniknięcia narzutu czasowego w pętlach
        self.acc_train = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.acc_val = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)

    def train_one_epoch(self, loader, epoch, num_epochs, fold):
        """
        Realizuje jedną epokę treningową z wykorzystaniem AMP i optymalizacji transferu danych.
        """
        self.model.train()
        self.acc_train.reset()  # Resetowanie statystyk metryki przed nową epoką
        running_loss = 0

        # Wykorzystanie 'paths' zgodnie ze strukturą ImageFolderWithPaths
        for data, targets, paths in tqdm(loader, desc=f"Fold {fold} Ep {epoch}/{num_epochs}"):
            # Transfer danych na GPU w trybie non_blocking dla lepszej utylizacji procesora
            data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

            # Przyspieszone zerowanie gradientów poprzez zwolnienie pamięci (set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)

            # Blok Autocast realizujący obliczenia w mieszanej precyzji (Mixed Precision)
            with torch.amp.autocast(device_type=self.device.type):
                scores = self.model(data)
                loss = self.criterion(scores, targets)

            # Skalowanie straty i wsteczna propagacja dla stabilności niskiej precyzji
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            # Obliczanie dokładności przy użyciu nowoczesnej składni argmax
            predicted = scores.argmax(dim=1)
            self.acc_train.update(predicted, targets)

        return running_loss / len(loader), self.acc_train.compute().item()

    def validate(self, loader):
        """
        Przeprowadza walidację modelu w trybie mieszanej precyzji bez obliczania gradientów.
        POPRAWKA: Usunięto zbędny argument num_classes, który powodował błąd wywołania.
        """
        self.model.eval()
        self.acc_val.reset()
        val_running_loss = 0

        # Wyłączenie śledzenia gradientów dla oszczędności VRAM
        with torch.no_grad():
            for images, labels, paths in loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                # Walidacja również w trybie AMP dla spójności z procesem treningowym
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                val_running_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                self.acc_val.update(predicted, labels)

        return val_running_loss / len(loader), self.acc_val.compute().item()