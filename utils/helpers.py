import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import os


def set_seed(seed=50):
    """
    Konfiguruje ziarno losowości dla wszystkich bibliotek składowych w celu
    zapewnienia pełnej deterministyczności i powtarzalności wyników treningu.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Blokada niedeterministycznych algorytmów biblioteki cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def show_augmented_samples(loader, class_names, num_images, config, save_path):
    """
    Generuje wizualizację próbek obrazów poddanych procesom augmentacji i normalizacji.
    Funkcja automatycznie odwraca proces normalizacji dla poprawnej prezentacji barw (RGB).
    """
    # Pobranie pierwszej paczki danych z obiektu DataLoader
    images, labels, _ = next(iter(loader))

    # Zabezpieczenie przed przekroczeniem rozmiaru batcha (limit 20 dla czytelności siatki)
    num_to_show = min(len(images), 20)

    # Dynamiczne wyznaczanie struktury siatki (maksymalnie 10 kolumn)
    cols = min(num_to_show, 10)
    rows = int(np.ceil(num_to_show / cols))

    plt.figure(figsize=(22, 3 * rows))

    for i in range(num_to_show):
        # Konwersja tensora do formatu NumPy i zmiana układu osi na (H, W, C)
        img = images[i].permute(1, 2, 0).numpy()

        # Denormalizacja obrazu na podstawie parametrów ze słownika CONFIG
        mean = np.array(config["norm_mean"])
        std = np.array(config["norm_std"])
        img = std * img + mean
        img = np.clip(img, 0, 1)  # Przycięcie wartości do zakresu poprawnego dla pyplot

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(class_names[labels[i]], fontsize=8)
        plt.axis('off')

    plt.suptitle(f"Podgląd potoku augmentacji ({num_to_show} próbek)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Weryfikacja i tworzenie struktury katalogów przed zapisem pliku
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    plt.show()
    plt.close()


def log_gradcam_paths(file_path, fold, successes, failures):
    """
    Archiwizuje ścieżki do obrazów wybranych do analizy metodą Grad-CAM.
    Zapisuje informacje o pewności predykcji (confidence) oraz etykietach rzeczywistych i przewidzianych.
    """
    # Tworzenie folderu nadrzędnego dla pliku logów, jeśli nie istnieje
    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Otwarcie pliku w trybie 'a' (append) w celu dopisania danych z kolejnego foldu
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n--- FOLD {fold} ---\n")

        f.write("TOP SUCCESSES (Próbki zaklasyfikowane poprawnie z najwyższą pewnością):\n")
        for s in successes:
            f.write(f"Path: {s['path']} | Conf: {s['confidence']:.4f}\n")

        f.write("TOP FAILURES (Próbki zaklasyfikowane błędnie z najwyższą pewnością):\n")
        for fail in failures:
            f.write(
                f"Path: {fail['path']} | Conf: {fail['confidence']:.4f} | True: {fail['true_idx']} | Pred: {fail['pred_idx']}\n")

        f.write("-" * 30 + "\n")