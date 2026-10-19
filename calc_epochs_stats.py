import os
import pandas as pd
import numpy as np

# Foldery wybrane do tego zadania
TARGET_FOLDERS = [
    "results/1_Final_testy_ptaki/FINAL_Ptaki_448px_CV5",
    "results/1_Final_testy_ptaki/EFFICIENCY_Ptaki_224px_CV5",
    "results/1_Final_testy_ptaki/BASELINE_Ptaki_Water_448px_CV5",
    "results/1_Final_testy_ptaki/SCALABILITY_CUB_Full_200klas",
    "results/1_Final_testy_samoloty/FINAL_Samoloty_448px_CV5",
    "results/1_Final_testy_samoloty/EFFICIENCY_Samoloty_224px_CV5",
    "results/1_Final_testy_samoloty/Drugi_Samoloty_CV5",
    "results/1_Final_testy_samoloty/SCALABILITY_Samoloty_Full_101klas",
    "results/Custom_Final_Ptaki_448px",
    "results/Custom_Final_Ptaki_224px",
    "results/Custom_Final_Ptaki_wodne",
    "results/Custom_Final_Ptaki_200_klas",
    "results/Custom_Final_Samoloty_448px",
    "results/Custom_Final_Samoloty_224px",
    "results/Custom_Final_Samoloty_drugie",
    "results/Custom_Final_Samoloty_200_klas",
]

# MAPOWANIE: Nazwa folderu -> Docelowa nazwa do wyprintowania
FOLDER_TO_NAME = {
    "FINAL_Ptaki_448px_CV5": "Ptaki 20 klas (448px) (EffNetB0)",
    "EFFICIENCY_Ptaki_224px_CV5": "Ptaki 20 klas (224px) (EffNetB0)",
    "BASELINE_Ptaki_Water_448px_CV5": "Ptaki Wodne 20 klas (EffNetB0)",
    "SCALABILITY_CUB_Full_200klas": "Ptaki 200 klas (EffNetB0)",
    "FINAL_Samoloty_448px_CV5": "Samoloty 20 klas (448px) (EffNetB0)",
    "EFFICIENCY_Samoloty_224px_CV5": "Samoloty 20 klas (224px) (EffNetB0)",
    "Drugi_Samoloty_CV5": "Samoloty drugie 20 klas (EffNetB0)",
    "SCALABILITY_Samoloty_Full_101klas": "Samoloty 101 klas (EffNetB0)",
    "Custom_Final_Ptaki_448px": "Ptaki 20 klas (448px) (MyCustomCNN)",
    "Custom_Final_Ptaki_224px": "Ptaki 20 klas (224px) (MyCustomCNN)",
    "Custom_Final_Ptaki_wodne": "Ptaki Wodne 20 klas (MyCustomCNN)",
    "Custom_Final_Ptaki_200_klas": "Ptaki 200 klas (MyCustomCNN)",
    "Custom_Final_Samoloty_448px": "Samoloty 20 klas (448px) (MyCustomCNN)",
    "Custom_Final_Samoloty_224px": "Samoloty 20 klas (224px) (MyCustomCNN)",
    "Custom_Final_Samoloty_drugie": "Samoloty drugie 20 klas (MyCustomCNN)",
    "Custom_Final_Samoloty_200_klas": "Samoloty 101 klas (MyCustomCNN)"
}


def get_epoch_stats():
    results = []

    for test_path in TARGET_FOLDERS:
        if not os.path.exists(test_path):
            print(f"[OSTRZEŻENIE] Brak folderu: {test_path}")
            continue

        folder_name = os.path.basename(test_path)
        clean_name = FOLDER_TO_NAME.get(folder_name, folder_name)

        epochs_from_folds = []

        for root, _, files in os.walk(test_path):
            for file in files:
                if file.endswith(".csv") and "_results" in file:
                    csv_path = os.path.join(root, file)
                    try:
                        # Wczytanie surowe, aby zignorować potencjalne problemy z nagłówkami
                        with open(csv_path, 'r') as f:
                            lines = f.readlines()

                        for line in lines:
                            if line.startswith("SUMMARY"):
                                parts = line.strip().split(',')
                                # Bierzemy twardy indeks nr 19, gdzie w logach zapisana jest liczba epok
                                best_ep = int(parts[19])
                                epochs_from_folds.append(best_ep)
                    except Exception as e:
                        print(f"Błąd czytania {csv_path}: {e}")

        if epochs_from_folds:
            avg_ep = np.mean(epochs_from_folds)
            min_ep = np.min(epochs_from_folds)
            max_ep = np.max(epochs_from_folds)

            results.append({
                "Name": clean_name,
                "Avg": avg_ep,
                "Min": min_ep,
                "Max": max_ep,
                "Folds": len(epochs_from_folds)
            })

    print("\n" + "=" * 70)
    print("STATYSTYKI EPOK (do zadziałania Early Stopping)")
    print("=" * 70)
    print(f"{'ZBIÓR (MODEL)':<40} | ŚREDNIA | MIN | MAX | POMIARY")
    print("-" * 70)

    for r in results:
        print(f"{r['Name']:<40} | {r['Avg']:<7.1f} | {r['Min']:<3} | {r['Max']:<3} | {r['Folds']} foldy")


if __name__ == "__main__":
    get_epoch_stats()