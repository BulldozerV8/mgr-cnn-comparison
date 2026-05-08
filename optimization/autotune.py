import optuna
import copy
import os
import sys

# 1. Pobierz absolutną ścieżkę
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Ustaw folder roboczy na root projektu
project_root = os.path.dirname(current_dir)
os.chdir(project_root)

# 3. Dodaj root projektu do ścieżek
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiment import ExperimentRunner
from config import CONFIG

# Testujemy wielkość budżetu epoki (Oversampling/Augmentacja)
SAMPLES_VARIANTS = [10000, 20000, 30000]


def objective(trial):
    # 1. Kopia bazowego configu
    local_config = copy.deepcopy(CONFIG)

    # 2. USTALONE PARAMETRY (Twoja "Złota Konfiguracja")
    local_config["batch_size"] = 128
    local_config["learning_rate"] = 0.0005
    local_config["unfreeze_blocks"] = 3
    local_config["head_width"] = 256
    local_config["head_depth"] = 2
    local_config["head_dropout"] = 0.2
    local_config["weight_decay"] = 0.0001
    local_config["pooling_mode"] = "avg"  # Powrót do bezpiecznego AVG

    # 3. DOSTRAJANIE LICZBY PRÓBEK PER EPOKA
    samples_val = trial.suggest_categorical("num_samples_per_epoch", SAMPLES_VARIANTS)
    local_config["num_samples_per_epoch"] = samples_val

    # 4. AUTOMATYCZNA NAZWA
    local_config["test_name"] = f"Tune_Samples_{samples_val}_Trial_{trial.number}"

    # 5. BLOKADY DLA SZYBKOŚCI
    local_config["use_cv"] = False
    local_config["use_gradcam"] = True
    local_config["num_epochs"] = 10
    local_config["early_stopping_patience"] = 3

    print(f"\n>>> TEST SKALI: NUM_SAMPLES_PER_EPOCH={samples_val}")

    try:
        runner = ExperimentRunner(local_config)
        score = runner.run_cross_validation()
        return score
    except Exception as e:
        print(f"Błąd Trial {trial.number}: {e}")
        return 0.0


if __name__ == "__main__":
    search_space = {"num_samples_per_epoch": SAMPLES_VARIANTS}
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(objective, n_trials=len(SAMPLES_VARIANTS))

    print("\n" + "=" * 30)
    print("WYNIK TESTU SKALI EPOKI:")
    print(study.best_params)
    print(f"Najlepszy wynik (Acc): {study.best_value:.4f}")
    print("=" * 30)