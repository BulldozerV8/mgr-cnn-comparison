import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================
# KONFIGURACJA STYLU AKADEMICKIEGO (ZACHOWANA 1:1)
# =================================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
OUTPUT_BASE = os.path.join("Thesis_Outputs", "5_4_custom")

# Foldery wybrane do tego zadania
TARGET_FOLDERS = [
    "1_Final_testy_samoloty/Drugi_Samoloty_CV5",
    "1_Final_testy_samoloty/SCALABILITY_Samoloty_Full_101klas",
    "Custom_Final_Ptaki_448px",
    "Custom_Final_Ptaki_224px",
    "Custom_Final_Ptaki_wodne",
    "Custom_Final_Ptaki_200_klas",
    "Custom_Final_Samoloty_448px",
    "Custom_Final_Samoloty_224px",
    "Custom_Final_Samoloty_drugie",
    "Custom_Final_Samoloty_200_klas"
]

def process_learning_curves():
    print("Generowanie podwójnych krzywych uczenia (Loss + Accuracy)...")
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    for test in TARGET_FOLDERS:
        folder_path = os.path.join("results", test)
        if not os.path.exists(folder_path):
            print(f"Pominięto, folder nie istnieje: {folder_path}")
            continue

        all_epochs = []
        test_name = os.path.basename(test)

        # Znajdź wszystkie pliki z wynikami w danym folderze i jego podfolderach
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".csv") and "_results" in file:
                    file_path = os.path.join(root, file)
                    try:
                        e_data = pd.read_csv(file_path)
                        if 'type' in e_data.columns:
                            e_data = e_data[e_data['type'] == 'EPOCH'].copy()

                            if not e_data.empty:
                                # Konwersja i skalowanie Accuracy (logika 1:1)
                                if e_data['train_acc'].max() <= 1.0: e_data['train_acc'] *= 100
                                if e_data['val_acc'].max() <= 1.0: e_data['val_acc'] *= 100

                                # Konwersja Loss na typ numeryczny
                                e_data['train_loss'] = pd.to_numeric(e_data['train_loss'], errors='coerce')
                                e_data['val_loss'] = pd.to_numeric(e_data['val_loss'], errors='coerce')
                                e_data['epoch'] = pd.to_numeric(e_data['epoch'], errors='coerce')

                                all_epochs.append(e_data)
                    except Exception as e:
                        print(f"Błąd czytania {file_path}: {e}")

        if not all_epochs:
            continue

        combined = pd.concat(all_epochs).dropna(subset=['epoch']).sort_values('epoch')

        # Agregacja średniej i odchylenia dla wszystkich 4 metryk
        stats = combined.groupby('epoch').agg({
            'train_loss': ['mean', 'std'], 'val_loss': ['mean', 'std'],
            'train_acc': ['mean', 'std'], 'val_acc': ['mean', 'std']
        }).reset_index()

        # Tworzenie siatki 1x2 (jeden rząd, dwie kolumny)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 1. WYKRES LEWY: Funkcja Kosztu (Loss)
        ax1.plot(stats['epoch'], stats['train_loss']['mean'], label='Trening (Loss)', color=COLORS[3],
                 marker='o')
        ax1.fill_between(stats['epoch'], stats['train_loss']['mean'] - stats['train_loss']['std'],
                         stats['train_loss']['mean'] + stats['train_loss']['std'], color=COLORS[3], alpha=0.15)
        ax1.plot(stats['epoch'], stats['val_loss']['mean'], label='Walidacja (Loss)', color=COLORS[4],
                 marker='o')
        ax1.fill_between(stats['epoch'], stats['val_loss']['mean'] - stats['val_loss']['std'],
                         stats['val_loss']['mean'] + stats['val_loss']['std'], color=COLORS[4], alpha=0.15)

        ax1.set_xlabel("Epoka")
        ax1.set_ylabel("Wartość Loss")
        ax1.set_title("Funkcja kosztu (Loss Curve)")
        ax1.set_xticks(range(int(stats['epoch'].min()), int(stats['epoch'].max()) + 1))
        ax1.set_ylim(bottom=0)  # Loss zaczyna się od zera w górę
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # 2. WYKRES PRAWY: Dokładność (Accuracy)
        ax2.plot(stats['epoch'], stats['train_acc']['mean'], label='Trening (Acc)', color=COLORS[0], marker='o')
        ax2.fill_between(stats['epoch'], stats['train_acc']['mean'] - stats['train_acc']['std'],
                         stats['train_acc']['mean'] + stats['train_acc']['std'], color=COLORS[0], alpha=0.15)
        ax2.plot(stats['epoch'], stats['val_acc']['mean'], label='Walidacja (Acc)', color=COLORS[1], marker='o')
        ax2.fill_between(stats['epoch'], stats['val_acc']['mean'] - stats['val_acc']['std'],
                         stats['val_acc']['mean'] + stats['val_acc']['std'], color=COLORS[1], alpha=0.15)

        ax2.set_xlabel("Epoka")
        ax2.set_ylabel("Dokładność (%)")
        ax2.set_title("Dokładność (Accuracy Curve)")
        ax2.set_xticks(range(int(stats['epoch'].min()), int(stats['epoch'].max()) + 1))
        ax2.set_ylim(0, 100)  # Twarda, naukowa skala 0-100%
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.5)

        # Dopasowanie układu i zapis
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_BASE, f"Krzywa_Uczenia_{test_name}.png"))
        plt.close()
        print(f"Wygenerowano: Krzywa_Uczenia_{test_name}.png")

if __name__ == "__main__":
    process_learning_curves()