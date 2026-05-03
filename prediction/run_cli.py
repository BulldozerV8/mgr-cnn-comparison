import os
from prediction.predictor import Predictor
from config import CONFIG

"""
Skrypt run_cli.py: Interfejs lini poleceń dla silnika predykcyjnego.
Służy do błyskawicznej weryfikacji skuteczności konkretnych wag modeli na pojedynczych 
próbkach danych lub całych katalogach bez konieczności uruchamiania pełnego interfejsu GUI.
"""

# --- 1. Konfiguracja zasobów i mapowanie ścieżek ---
# System automatycznie mapuje lokalizację wag na podstawie nazwy eksperymentu z CONFIG.
# Pozwala to na zachowanie spójności między fazą treningu a fazą testów operacyjnych.
TEST_NAME = CONFIG["test_name"]
TEST_FOLDER = os.path.join("results", TEST_NAME)

# Definicja ścieżki do konkretnego artefaktu modelu (np. najlepszy model z pierwszego folda)
MODEL_PATH = os.path.join(TEST_FOLDER, "saved_models", "model_weights_fold_1.pth")
IMAGE_PATH = "test_image.jpg"  # Lokalizacja obrazu wejściowego do analizy

# --- 2. Walidacja dostępności zasobów (Pre-flight Checks) ---
# Weryfikacja fizycznego istnienia plików przed alokacją zasobów GPU/VRAM.
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Błąd krytyczny: Nie odnaleziono wag modelu w lokalizacji: {MODEL_PATH}")

if not os.path.exists(IMAGE_PATH):
    print(f"Błąd: Plik wejściowy {IMAGE_PATH} nie istnieje. Prerywanie operacji.")
    exit()

# --- 3. Inicjalizacja autonomicznego silnika Predictor ---
# Silnik automatycznie wczytuje metadane (model_info.json) oraz mapę klas (class_names.json)
# znajdujące się w folderze nadrzędnym wag, co eliminuje potrzebę ręcznej konfiguracji.
predictor = Predictor(MODEL_PATH, device=CONFIG["device"])

# --- 4. Egzekucja procesu inferencji ---
# Wywołanie analizy z parametrem Top-K określającym liczbę zwracanych najbardziej prawdopodobnych klas.
print(f"\n{'=' * 20} ANALIZA OBRAZU: {os.path.basename(IMAGE_PATH)} {'=' * 20}")
result = predictor.predict(IMAGE_PATH, top_k=3)

# --- 5. Prezentacja wyników i raportowanie (Console Output) ---
# Przetwarzanie ustrukturyzowanego słownika wynikowego zwróconego przez Predictor.
if result["status"] == "success":
    print(f"Wydajność systemu: {result['time_ms']:.2f} ms")
    print("-" * 50)
    print(f"{'POZYCJA':<10} | {'KLASA ROZPOZNANA':<25} | {'PEWNOŚĆ (%)'}")
    print("-" * 50)

    for i, res in enumerate(result["top_k"]):
        label = res['label']
        confidence = res['confidence'] * 100
        print(f"{i + 1:<10} | {label:<25} | {confidence:>10.2f}%")

    print("-" * 50)
    print(f"Status: Sukces. Pełny raport został zarchiwizowany w logach sesji.")
else:
    # Obsługa błędów procesowych (np. problem z dekodowaniem obrazu)
    print(f"STATUS BŁĘDU: {result.get('message')}")

# --- OPCJONALNIE: Przetwarzanie wsadowe (Batch Processing) ---
# Odkomentowanie poniższej linii uruchamia masową analizę wszystkich obrazów w folderze.
# predictor.predict_folder("sciezka/do/katalogu_z_danymi")