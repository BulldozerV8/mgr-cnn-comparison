import os
import pandas as pd

# Ścieżka do Twojego folderu results
BASE_DIR = "../results"
# Tylko te foldery będą przeszukiwane:
TARGET_FOLDERS = [
    "1_Final_testy_ptaki",
    "1_Final_testy_samoloty",
    "1_Final_tuning_ptaki",
    "1_Final_tuning_samoloty"
]
OUTPUT_FILE = "master_manifest.csv"

all_summaries = []

print(f"Rozpoczynam zbieranie danych z wybranych folderów w {BASE_DIR}...")

for target in TARGET_FOLDERS:
    target_path = os.path.join(BASE_DIR, target)

    if not os.path.exists(target_path):
        print(f"Pominięto: {target} (folder nie istnieje)")
        continue

    print(f"Przeszukuję: {target}...")

    for root, dirs, files in os.walk(target_path):
        for file in files:
            # Szukamy plików wyników
            if file.endswith(".csv") and "_results" in file:
                file_path = os.path.join(root, file)
                try:
                    # TYLKO ODCZYT - brak zmian w plikach oryginalnych
                    df = pd.read_csv(file_path)

                    # Szukamy wiersza typu SUMMARY
                    summary_data = df[df['type'] == 'SUMMARY']

                    # Jeśli brak SUMMARY (tuning/pojedynczy test), bierzemy ostatnią epokę
                    if summary_data.empty:
                        summary_data = df[df['type'] == 'EPOCH'].tail(1)

                    if not summary_data.empty:
                        row = summary_data.iloc[0].to_dict()
                        # Zapisujemy skąd pochodzi wynik dla łatwej identyfikacji
                        row['folder_group'] = target
                        row['subfolder'] = os.path.relpath(root, target_path)
                        row['file_source'] = file
                        all_summaries.append(row)

                except Exception as e:
                    print(f"Błąd w pliku {file_path}: {e}")

if all_summaries:
    master_df = pd.DataFrame(all_summaries)
    # Porządkujemy kolumny: grupa, podfolder i nazwa testu na początek
    meta_cols = ['folder_group', 'subfolder', 'test_name', 'dataset_tag']
    other_cols = [c for c in master_df.columns if c not in meta_cols]
    master_df = master_df[meta_cols + other_cols]

    master_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nSukces! Zebrano dane z {len(all_summaries)} testów.")
    print(f"Wynik zapisany w nowym pliku: {OUTPUT_FILE}")
else:
    print("\nNie znaleziono danych spełniających kryteria w wybranych folderach.")