import optuna
import copy
import os
import time  # DODANE: do mierzenia czasu treningu
import optuna.visualization as vis
from experiment import ExperimentRunner
from config import CONFIG


def objective(trial):
    # 1. KOPIA TWOJEGO ORYGINALNEGO CONFIGU
    local_config = copy.deepcopy(CONFIG)

    # ==========================================================
    # === SUWAKI (TO CO CHCESZ TERAZ TESTOWAĆ) ===
    # ==========================================================
    # Na razie sprawdzamy tylko batch_size (zgodnie z planem: 32, 64, 128, 256)
    local_config["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    # ==========================================================

    # === BLOKADY TECHNICZNE (USTAWIONE NA SZTYWNO DLA TUNERA) ===
    local_config["robustness_mode"] = False  # Musimy trenować, nie tylko testować
    local_config["use_cv"] = False  # 1 fold dla szybkości
    local_config["use_gradcam"] = False  # Wyłączone, by nie śmiecić obrazkami
    local_config["use_seed"] = True  # Rzetelne porównanie wyników
    local_config["test_name"] = f"AutoTune_Batch_{local_config['batch_size']}_Trial_{trial.number}"

    start_t = time.time()  # START STOPER
    try:
        # Odpalenie Twojego runnera
        runner = ExperimentRunner(local_config)
        # Zwracamy średnią skuteczność do Optuny
        score = runner.run_cross_validation()
        duration = time.time() - start_t  # OBLICZ CZAS

        return score, duration  # ZWRACAMY ACC I CZAS
    except Exception as e:
        print(f"Błąd podczas Trial {trial.number}: {e}")
        return 0.0, 99999.0


if __name__ == "__main__":
    # Inicjalizacja badania (maximize dla Acc, minimize dla Czasu)
    study = optuna.create_study(directions=["maximize", "minimize"])

    # Wykonujemy np. 5 prób, żebyś zobaczył jak to działa
    study.optimize(objective, n_trials=4)

    # === SEKCJA GENEROWANIA WYKRESÓW ZBIORCZYCH (DODANE) ===
    # Używamy os.path.dirname, aby folder powstał obok CSV, a nie "wewnątrz" nazwy pliku
    plot_dir = os.path.join(CONFIG["results_path"], "optuna_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Wykres: Batch Size vs Celność (Z polskim opisem)
    fig_acc = vis.plot_slice(study, target=lambda t: t.values[0], target_name="Celność (Accuracy)")
    fig_acc.update_layout(title="Porównanie: Batch Size vs Celność")
    fig_acc.write_image(os.path.join(plot_dir, "wynik_celnosc.png"))

    # 2. Wykres: Batch Size vs Czas (To co mówiłeś, że ważne)
    fig_time = vis.plot_slice(study, target=lambda t: t.values[1], target_name="Czas treningu (s)")
    fig_time.update_layout(title="Porównanie: Batch Size vs Czas treningu")
    fig_time.write_image(os.path.join(plot_dir, "wynik_czas.png"))

    # 3. Wykres Pareto (Pokazuje kompromis między szybkością a jakością)
    fig_pareto = vis.plot_pareto_front(study, target_names=["Celność", "Czas treningu"])
    fig_pareto.update_layout(title="Front Pareto: Celność vs Czas")
    fig_pareto.write_image(os.path.join(plot_dir, "pareto_front.png"))
    # ======================================================

    print("\n" + "=" * 40)
    print("NAJLEPSZE WYNIKI (KOMPROMIS ACC/CZAS):")
    for trial in study.best_trials:
        print(
            f"Trial {trial.number}: Batch={trial.params['batch_size']}, Acc={trial.values[0]:.4f}, Czas={trial.values[1]:.2f}s")
    print("=" * 40)