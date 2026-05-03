import sys
from experiment import ExperimentRunner
from config import CONFIG

"""
Główny punkt wejścia systemu (System Entry Point).
Skrypt odpowiada za hierarchiczną inicjalizację zasobów, zarządzanie cyklem życia 
eksperymentu oraz globalną obsługę wyjątków w celu zapewnienia stabilności 
długotrwałych procesów obliczeniowych.
"""

def main():
    # --- 1. Weryfikacja środowiska operacyjnego ---
    # Raportowanie stanu sprzętowego przed alokacją zasobów GPU/VRAM.
    print(f"Wykryte urządzenie: {CONFIG['device']}")

    try:
        # --- 2. Inicjalizacja i egzekucja rurociągu badawczego ---
        # Orkiestracja procesów: przekazanie konfiguracji do silnika wykonawczego.
        runner = ExperimentRunner(CONFIG)
        runner.run_cross_validation()

    except KeyboardInterrupt:
        # --- 3. Procedura bezpiecznego zamknięcia (Graceful Exit) ---
        # Obsługa przerwań systemowych pozwalająca na kontrolowane zwolnienie
        # zasobów i uniknięcie blokady pamięci operacyjnej urządzenia.
        print("\n[!] Operacja przerwana przez użytkownika (KeyboardInterrupt).")
        sys.exit(0)

    except Exception as e:
        # --- 4. Mechanizm odporności na błędy (Fault Tolerance) ---
        # Przechwytywanie błędów krytycznych w celu wygenerowania raportu technicznego
        # bez niekontrolowanego zamknięcia sesji terminala.
        print(f"\n[BŁĄD KRYTYCZNY SYSTEMU]: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()