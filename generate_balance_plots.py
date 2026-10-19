import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter

# =================================================================
# KONFIGURACJA STYLU I ŚRODOWISKA
# =================================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

SEED = 50
OUTPUT_DIR = os.path.join("Thesis_Outputs", "Wykresy_Balansowania")

# Definicje zbiorów danych na podstawie ustaleń
DATASETS = [
    {
        "name": "Samoloty_20_klas",
        "path": "datasets/Aircraft_close",
        "limit": 30000
    },
    {
        "name": "Ptaki_20_klas",
        "path": "datasets/CUB-2",
        "limit": 30000
    },
    {
        "name": "Samoloty_101_klas",
        "path": "datasets/Aircraft-full",
        "limit": 50000  # Wyższy limit dla ekstremalnie niezbalansowanego, pełnego zbioru
    },
    {
        "name": "Ptaki_200_klas",
        "path": "datasets/CUB-full",
        "limit": 30000
    }
]


def generate_balance_plots():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Rozpoczynam symulację balansowania klas (Oversampling)...\n")

    for ds_info in DATASETS:
        ds_path = ds_info["path"]

        if not os.path.exists(ds_path):
            print(f"[POMINIĘTO] Brak folderu: {ds_path}")
            continue

        print(f"Przetwarzanie: {ds_info['name']}...")

        # 1. Wczytanie struktury danych
        dataset = datasets.ImageFolder(ds_path)
        targets = np.array(dataset.targets)
        class_names = dataset.classes

        # 2. Symulacja podziału dokładnie tak, jak robi to ExperimentRunner
        indices = np.arange(len(targets))

        # Odcięcie 15% na zbiór testowy
        cv_indices, _ = train_test_split(
            indices, test_size=0.15, shuffle=True, stratify=targets, random_state=SEED
        )
        cv_targets = targets[cv_indices]

        # Walidacja krzyżowa (5 foldów) -> bierzemy indeksy pierwszego folda
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_fold_idx, _ = next(skf.split(cv_indices, cv_targets))

        # Fizyczne indeksy i etykiety zdjęć treningowych
        train_indices = cv_indices[train_fold_idx]
        train_labels = targets[train_indices]

        # 3. Zliczanie zdjęć PRZED balansowaniem
        count_before = Counter(train_labels)

        # 4. Logika ważenia identyczna z DataManager
        _, class_sample_count = np.unique(train_labels, return_counts=True)
        weight = 1. / class_sample_count
        samples_weight = torch.from_numpy(np.array([weight[t] for t in train_labels]))

        # Tworzenie samplera
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=samples_weight,
            num_samples=ds_info["limit"],
            replacement=True
        )

        # Wymuszamy ziarno losowości PyTorch, aby losowanie było deterministyczne
        torch.manual_seed(SEED)

        # Generujemy losowe indeksy bezpośrednio z samplera
        sampled_indices = list(sampler)
        sampled_labels = [train_labels[i] for i in sampled_indices]

        # 5. Zliczanie zdjęć PO balansowaniu
        count_after = Counter(sampled_labels)

        # 6. Tworzenie połączonego wykresu (Subplots 1x2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Sortowanie klas alfabetycznie (zgodnie z macierzą pomyłek)
        sorted_classes = sorted(range(len(class_names)), key=lambda x: class_names[x])

        before_vals = [count_before.get(i, 0) for i in sorted_classes]
        after_vals = [count_after.get(i, 0) for i in sorted_classes]
        x_labels = [class_names[i] for i in sorted_classes]
        x_pos = np.arange(len(sorted_classes))

        # WYKRES LEWY (Przed)
        ax1.bar(x_pos, before_vals, color='#1f77b4', edgecolor='black', linewidth=0.5)
        ax1.set_title("Przed balansowaniem", fontweight='bold', pad=10)
        ax1.set_ylabel("Liczba zdjęć", fontweight='bold')

        # WYKRES PRAWY (Po)
        ax2.bar(x_pos, after_vals, color='#ff7f0e', edgecolor='black', linewidth=0.5)
        ax2.set_title("Po balansowaniu", fontweight='bold', pad=10)
        ax2.set_ylabel("Liczba zdjęć", fontweight='bold')

        # Dopasowanie osi X w zależności od liczby klas
        if len(class_names) <= 20:
            # Dla małych zbiorów wyświetlamy nazwy klas pod kątem, usuwamy napis "Klasy"
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(x_labels, rotation=90, fontsize=9)
            ax1.set_xlabel("")

            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(x_labels, rotation=90, fontsize=9)
            ax2.set_xlabel("")
        else:
            # Dla dużych zbiorów (101/200 klas) wyłączamy podpisy by nie tworzyć czarnej plamy, wstawiamy napis "Klasy"
            ax1.set_xticks([])
            ax1.set_xlabel("Klasy", fontweight='bold', labelpad=10)

            ax2.set_xticks([])
            ax2.set_xlabel("Klasy", fontweight='bold', labelpad=10)

        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, f"Balansowanie_{ds_info['name']}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"--> Zapisano: {save_path}")

    print("\nZakończono. Wszystkie wykresy są w folderze Thesis_Outputs/Wykresy_Balansowania.")


if __name__ == "__main__":
    generate_balance_plots()