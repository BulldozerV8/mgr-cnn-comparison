import os
import sys
import copy

# 1. Konfiguracja ścieżek systemowych w celu umożliwienia importów z katalogu głównego

project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiment import ExperimentRunner
from config import CONFIG

# Definicja dwóch podzbiorów do przetestowania (nazwa testu oraz ścieżka do folderu)
EVALUATION_JOBS = [
    {
        "test_name": "SCALABILITY_Samoloty_Full_101klas",
        "dataset_tag": "Aircraft-full",
        "data_path": "datasets/Aircraft-full/"
    },
    {
        "test_name": "Drugi_Samoloty_CV5",
        "dataset_tag": "Aircraft_close_second",
        "data_path": "datasets/Aircraft_close_second/"
    }
]

if __name__ == "__main__":
    print("=== WSTĘPNA WERYFIKACJA ŚCIEŻEK ===")
    all_paths_ok = True

    for job in EVALUATION_JOBS:
        if not os.path.exists(job["data_path"]):
            print(f"[BŁĄD KRYTYCZNY] Ścieżka '{job['data_path']}' dla testu '{job['test_name']}' NIE ISTNIEJE!")
            all_paths_ok = False
        else:
            print(f"[OK] Znaleziono folder dla: {job['test_name']}")

    if not all_paths_ok:
        print("\nPrzerwano uruchomienie. Popraw ścieżki na dysku przed startem!")
        sys.exit(1)

    print("===================================\n")
    print(f"Rozpoczęcie sekwencyjnego uruchamiania testów końcowych")

    for job in EVALUATION_JOBS:
        print("\n" + "=" * 70)
        print(f"URUCHAMIANIE EKSPERYMENTU: {job['test_name']}")
        print(f"ŚCIEŻKA DO DANYCH: {job['data_path']}")
        print("=" * 70 + "\n")

        # Głęboka kopia domyślnego CONFIG (bierze Twoje bazowe parametry)
        local_config = copy.deepcopy(CONFIG)

        # Nadpisywanie parametrów operacyjnych pod dany bieg
        local_config["test_name"] = job["test_name"]
        local_config["dataset_tag"] = job["dataset_tag"]
        local_config["data_path"] = job["data_path"]

        # Dynamiczne skalowanie liczby próbek dla pełnego zbioru
        if job["test_name"] == "SCALABILITY_Samoloty_Full_101klas":
            local_config["num_samples_per_epoch"] = 50000
            print("-> Wykryto pełen zbiór danych: zwiększono 'num_samples_per_epoch' do 50000.")
        else:
            print(
                f"-> Podzbiór standardowy: zachowano domyślne 'num_samples_per_epoch' ({local_config['num_samples_per_epoch']}).")

        # Wymuszenie pełnej walidacji krzyżowej i domyślnych blokad epok z CONFIG
        local_config["use_cv"] = True
        local_config["use_gradcam"] = True

        try:
            # Inicjalizacja i odpalenie pełnej walidacji (5 foldów po 40 epok)
            runner = ExperimentRunner(local_config)
            final_score = runner.run_cross_validation()
            print(f"\n[SUKCES] Zakończono {job['test_name']}. Średni wynik: {final_score:.4f}")

        except Exception as e:
            print(f"\n[BŁĄD KRYTYCZNY] Eksperyment {job['test_name']} nie powiódł się: {e}")

    print("\n" + "=" * 70)
    print("ZAKOŃCZONO WSZYSTKIE ZAPLANOWANE EKSPERYMENTY")
    print("=" * 70)