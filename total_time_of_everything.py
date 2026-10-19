import os

# Ścieżka do Twojego głównego folderu z wynikami
root_dir = "results"

total_seconds = 0.0
csv_count = 0

# Przeszukiwanie wszystkich folderów i podfolderów
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".csv"):
            csv_count += 1
            filepath = os.path.join(dirpath, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    for line in file:
                        if line.startswith("SUMMARY"):
                            # Dzielimy linię po przecinku, usuwamy ewentualne spacje i bierzemy ostatni element
                            parts = line.strip().split(',')
                            if parts:
                                last_val = parts[-1]
                                try:
                                    val_float = float(last_val)
                                    total_seconds += val_float

                                    # Wypisanie wartości tylko dla konkretnego pliku
                                    if filename == "FINAL_Ptaki_448px_CV5_results.csv":
                                        print(f"Znaleziono w {filename} -> dodaję: {val_float}")

                                except ValueError:
                                    print(f"Nie można przekonwertować wartości '{last_val}' w pliku {filepath}")
            except Exception as e:
                print(f"Nie można otworzyć pliku {filepath}: {e}")

# Konwersja na godziny
total_hours = total_seconds / 3600

print("\n--- RAPORT ---")
print(f"Przeszukano plików CSV: {csv_count}")
print(f"Łączny czas (sekundy): {total_seconds:.2f} s")
print(f"Łączny czas (godziny): {total_hours:.2f} h")