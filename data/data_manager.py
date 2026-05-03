import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
from data.transforms import TransformFactory


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Rozszerzenie klasy ImageFolder o zwracanie ścieżki do pliku graficznego.
    Pozwala na powiązanie wyniku predykcji z konkretnym plikiem na dysku,
    co jest niezbędne do poprawnego działania wizualizacji Grad-CAM.
    """

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path


class DataManager:
    """
    Moduł zarządzający potokiem danych. Odpowiada za ładowanie zdjęć,
    aplikowanie transformacji oraz przygotowanie loaderów z uwzględnieniem
    balansowania liczności klas (oversampling).
    """

    def __init__(self, config):
        self.config = config

    def get_fold_loaders(self, train_idx, val_idx, test_idx):
        """
        Generuje loadery dla konkretnego foldu walidacji krzyżowej.
        Stosuje WeightedRandomSampler, aby każda klasa (nawet rzadka)
        miała taką samą wagę podczas procesu uczenia.
        """
        train_transforms = TransformFactory.get_train_transforms(self.config)
        test_transforms = TransformFactory.get_test_transforms(self.config)

        dataset_train_base = ImageFolderWithPaths(self.config["data_path"], transform=train_transforms)
        dataset_val_base = ImageFolderWithPaths(self.config["data_path"], transform=test_transforms)
        dataset_test_base = ImageFolderWithPaths(self.config["data_path"], transform=test_transforms)

        train_subset = Subset(dataset_train_base, train_idx)
        val_subset = Subset(dataset_val_base, val_idx)
        test_subset = Subset(dataset_test_base, test_idx)

        # --- LOGIKA BALANSOWANIA KLAS (OVERSAMPLING) ---

        targets = np.array(dataset_train_base.targets)
        train_labels = targets[train_idx]

        # Obliczanie wag próbek na podstawie liczności klas w zbiorze treningowym
        _, class_sample_count = np.unique(train_labels, return_counts=True)
        weight = 1. / class_sample_count
        samples_weight = torch.from_numpy(np.array([weight[t] for t in train_labels]))

        # Sampler losuje zdjęcia z prawdopodobieństwem odwrotnie proporcjonalnym do liczności klasy
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=samples_weight,
            num_samples=self.config["num_samples_per_epoch"],
            replacement=True
        )

        # --- INICJALIZACJA DATALOADERÓW ---

        train_loader = DataLoader(
            train_subset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            num_workers=self.config["num_workers"],
            pin_memory=True,  # Przyspiesza przesyłanie danych do pamięci GPU
            persistent_workers = True  # Utrzymuje procesy ładowania danych między epokami (szybszy start epoki)
        )

        val_loader = DataLoader(val_subset, batch_size=self.config["batch_size"], shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=self.config["batch_size"], shuffle=False)

        return train_loader, val_loader, test_loader