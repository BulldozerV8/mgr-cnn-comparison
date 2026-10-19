import optuna
import copy
import os
import sys
import torch

# 1. Konfiguracja ścieżek systemowych w celu umożliwienia importów z katalogu głównego
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiment import ExperimentRunner
from config import CONFIG

# ==============================================================================
# SEKCJA KONFIGURACJI TESTU (TUTAJ ZMIENIASZ PARAMETRY)
# ==============================================================================
# Nazwa parametru musi być identyczna z kluczem w słowniku CONFIG
TUNE_PARAMS = {
    "num_samples_per_epoch": [10000,20000]
    # Możesz dodać więcej, np.: "learning_rate": [0.001, 0.0005]
}

# Parametry "Złotego Standardu" - zostaną nadpisane w CONFIG przed testem
GOLDEN_CONFIG = {
    "batch_size": 16,
    "learning_rate": 0.0001,
    "custom_base_channels": 32,
    "custom_block_repeats": [3, 3, 4],
    "expansion_ratio": 4,
    "use_depthwise": True,
    "kernel_size": 5,
    "head_width": 256,
    "head_depth": 3,
    "drop_path_rate": 0.2,
    "pooling_mode": "avg",
    "head_dropout": 0.3,
    "weight_decay": 1e-04,

}
# ==============================================================================

def objective(trial):
    """
    Funkcja celu dla biblioteki Optuna. Realizuje pojedynczą próbę (Trial)
    eksperymentalną z dynamicznie generowaną konfiguracją.
    """
    # Głęboka kopia bazowej konfiguracji w celu uniknięcia interferencji między próbami
    local_config = copy.deepcopy(CONFIG)

    # Aplikacja ustalonych parametrów optymalnych (Golden Standard)
    for param, value in GOLDEN_CONFIG.items():
        local_config[param] = value

    # Dynamiczne sugerowanie parametrów i budowa nazwy testu
    test_suffix = ""
    for param_name, variants in TUNE_PARAMS.items():
        chosen_val = trial.suggest_categorical(param_name, variants)
        local_config[param_name] = chosen_val
        test_suffix += f"_{param_name}_{chosen_val}"

    # --- DODAJ TE 3 LINIKI: ---
    #if local_config.get("head_dropout") == 0.3 and local_config.get("weight_decay") == 1e-04:
    #    print(">>> POMIJAM ZNANĄ KOMBINACJĘ: 0.3 + 1e-04")
    #    raise optuna.TrialPruned()
    # --------------------------
    # Konfiguracja parametrów operacyjnych dla trybu szybkiej weryfikacji
    local_config["test_name"] = f"Tune{test_suffix}_Trial_{trial.number}"
    local_config["use_cv"] = False
    local_config["use_gradcam"] = True
    local_config["num_epochs"] = 20
    local_config["early_stopping_patience"] = 5

    print(f"\n>>> ROZPOCZĘCIE PRÓBY {trial.number}: {test_suffix[1:]}")

    try:
        # Inicjalizacja orkiestratora i wykonanie pętli badawczej
        runner = ExperimentRunner(local_config)
        score = runner.run_cross_validation()
        return score
    except Exception as e:
        print(f"Błąd krytyczny podczas Trial {trial.number}: {e}")
        return 0.0

if __name__ == "__main__":
    """
    Punkt wejścia skryptu. Inicjalizuje proces optymalizacji z wykorzystaniem GridSampler
    w celu przeszukania pełnej przestrzeni zdefiniowanych wariantów.
    """
    # Obliczanie całkowitej liczby kombinacji do sprawdzenia
    n_trials = 1
    for v in TUNE_PARAMS.values():
        n_trials *= len(v)

    # Inicjalizacja badania z deterministycznym próbkowaniem po siatce (Grid Search)
    sampler = optuna.samplers.GridSampler({k: v for k, v in TUNE_PARAMS.items()})
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(objective, n_trials=n_trials)

    # Raportowanie końcowe wyników optymalizacji
    print("\n" + "="*50)
    print("ZAKOŃCZONO PROCES OPTYMALIZACJI")
    print(f"Najlepsze parametry: {study.best_params}")
    print(f"Najwyższa uzyskana dokładność (Acc): {study.best_value:.4f}")
    print("="*50)