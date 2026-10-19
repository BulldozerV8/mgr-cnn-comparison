import numpy as np
import torch
import psutil
from torch import optim, nn
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import time
import json

from data.data_manager import DataManager
from models.cnn_base import CNN
from utils.logger import CSVLogger
from training.callbacks import EarlyStopping
from evaluation.evaluator import ModelEvaluator
from training.training_engine import TrainingEngine
from evaluation.visualization import Visualizer
from utils.helpers import set_seed, show_augmented_samples, log_gradcam_paths


class ExperimentRunner:
    """
    Główna klasa orkiestrująca (Orchestrator) proces badawczy.
    Odpowiada za spięcie wszystkich modułów: przygotowania danych, konfiguracji modelu,
    pętli treningowej (w tym walidacji krzyżowej), ewaluacji oraz zarządzania pamięcią GPU.
    """

    def __init__(self, config):
        """
        Inicjalizuje główne parametry eksperymentu i przygotowuje środowisko robocze.
        """
        self.config = config
        self._setup_paths()

    def _setup_paths(self):
        """
        Tworzy hermetyczną strukturę katalogów dla danego eksperymentu.
        Gwarantuje, że logi, wagi modeli, wykresy i mapy ciepła są grupowane
        w jednym spójnym folderze roboczym (test_dir).
        """

        base_results_dir = self.config.get("results_path", "results")

        # Główny folder testu: results/EffNet_Baseline_V1/
        self.test_dir = os.path.join(base_results_dir, self.config["test_name"])
        os.makedirs(self.test_dir, exist_ok=True)

        # 1. Ścieżka do CSV (dynamiczna)
        self.config["results_path"] = os.path.join(self.test_dir, f"{self.config['test_name']}_results.csv")

        # 2. Ścieżka do wag modeli
        self.config["save_dir"] = os.path.join(self.test_dir, "saved_models")
        os.makedirs(self.config["save_dir"], exist_ok=True)

        # 3. Ścieżki Grad-CAM
        self.config["gradcam_txt_path"] = os.path.join(self.test_dir, f"{self.config['test_name']}_gradcam_log.txt")
        self.gradcam_img_dir = os.path.join(self.test_dir, "gradcam_images")

        # 4. Ścieżka do wykresów
        self.plots_dir = os.path.join(self.test_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def run_cross_validation(self):
        """
        Główna metoda uruchamiająca proces badawczy. Zarządza ziarnem losowości,
        podziałem zestawu danych i agreguje końcowe wyniki z poszczególnych foldów.
        """
        # --- SEED I START ---
        if self.config["use_seed"]:
            set_seed(self.config["seed"])
            print(f"Ziarno ustawione na: {self.config['seed']}")
        else:
            print("Ziarno wyłączone - pełna losowość systemu.")

        # Optymalizacja CUDNN dla stałych rozmiarów zdjęć (przyspiesza operacje konwolucyjne)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        logger = CSVLogger(self.config["results_path"])
        start_total = time.time()

        # --- PRZYGOTOWANIE INDEKSÓW ---
        temp_dataset = datasets.ImageFolder(self.config["data_path"])
        targets = np.array(temp_dataset.targets)
        indices = np.arange(len(targets))
        class_names = temp_dataset.classes
        num_classes = len(class_names)

        cv_indices, test_idx = train_test_split(
            indices, test_size=self.config["test_size"], shuffle=True,
            stratify=targets, random_state=self.config["seed"] if self.config["use_seed"] else None
        )
        cv_targets = targets[cv_indices]

        # --- WYBÓR TRYBU (CV vs Simple) ---
        folds_to_run = self._get_folds_logic(cv_indices, cv_targets)
        cv_scores = []

        # --- PĘTLA PO FOLDACH ---
        for fold, (train_fold_idx, val_fold_idx) in enumerate(folds_to_run):
            fold_num = fold + 1
            print(f"\n{'=' * 20} FOLD {fold_num}/{len(folds_to_run)} {'=' * 20}")

            # Wywołujemy pojedynczy fold
            acc = self._run_single_fold(fold_num, cv_indices, train_fold_idx, val_fold_idx,
                                        test_idx, class_names, num_classes, logger)

            # POPRAWKA: Konwersja na float zapobiega błędom torch.mean przy logowaniu końcowym
            cv_scores.append(float(acc))

        # --- KONIEC EKSPERYMENTU ---
        self._finalize_experiment(start_total, cv_scores, class_names)

        # Kluczowe dla Optuny: zwracamy średnią skuteczność ze wszystkich foldów
        return float(np.mean(cv_scores))

    def _get_folds_logic(self, cv_indices, cv_targets):
        """
        Definiuje strategię podziału danych: wielokrotna walidacja krzyżowa (K-Fold)
        lub pojedynczy podział Hold-out (szybki tryb testowy).
        """
        if self.config["use_cv"]:
            skf = StratifiedKFold(n_splits=self.config["num_folds"], shuffle=True,
                                  random_state=self.config["seed"] if self.config["use_seed"] else None)
            print(f"TRYB: Walidacja Krzyżowa ({self.config['num_folds']} foldów)")
            return list(skf.split(cv_indices, cv_targets))
        else:
            t_idx, v_idx = train_test_split(
                np.arange(len(cv_indices)), test_size=self.config["val_size"],
                stratify=cv_targets, random_state=self.config["seed"] if self.config["use_seed"] else None
            )
            print(f"TRYB: Szybki (Tylko 1 fold - CV wyłączone)")
            return [(t_idx, v_idx)]

    def _run_single_fold(self, fold_num, cv_indices, train_idx, val_idx, test_idx, class_names, num_classes, logger):
        """
        Przeprowadza pełny cykl życia dla pojedynczego podziału danych (folda).
        Inicjalizuje model, wykonuje trening, wymusza ewaluację i rygorystycznie czyści pamięć.
        """
        start_fold_total = time.time()

        # --- DYNAMICZNY LIMIT PRÓBEK (MULTIPLIKATOR) ---
        N = len(train_idx)  # Liczba fizycznych zdjęć w tym foldzie

        # Pobieramy mnożnik z configu (domyślnie 1, żeby nic nie zmieniać w 10-klasowych testach)
        multiplier = self.config.get("samples_multiplier", 1)
        base_limit = self.config.get("num_samples_per_epoch", 30000)

        # Jeśli masz flagę włączoną, liczysz mnożnik, jeśli nie - zostaje baza 30k
        if multiplier > 1:
            self.config["num_samples_per_epoch"] = max(base_limit, int(N * multiplier))
        # -----------------------------------------------

        # 1. Loadery
        data_module = DataManager(self.config)
        train_loader, val_loader, test_loader = data_module.get_fold_loaders(
            cv_indices[train_idx], cv_indices[val_idx], test_idx
        )

        # 2. Aircraft-full Info (tylko w 1. foldzie)
        if fold_num == 1:
            print(f"Wykryto klas: {num_classes}")
            print(f"Zbiór fizyczny (train): {len(train_idx)} zdjęć")
            print(f"Zbiór efektywny (per epoch): {self.config['num_samples_per_epoch']} zdjęć")
            # Zapis podglądu augmentacji
            aug_save_path = os.path.join(self.plots_dir, "augmentation_preview.png")
            show_augmented_samples(train_loader, class_names, 20, self.config, aug_save_path)

        # 3. Model i Optimizer
        model = CNN(
            num_classes=num_classes,
            **self.config
        ).to(self.config["device"])

        criterion = nn.CrossEntropyLoss(label_smoothing=self.config["label_smoothing"])
        params_count = sum(p.numel() for p in model.parameters())

        # W experiment.py (fragment w _run_single_fold)
        if self.config.get("differential_lr", False):
            # Grupowanie parametrów: głowica uczy się szybciej, baza wolniej
            params = [
                {'params': model.model.features.parameters(), 'lr': self.config["learning_rate"] * 0.1},  # Baza
                {'params': model.model.classifier.parameters(), 'lr': self.config["learning_rate"]}  # Głowica
            ]
        else:
            params = model.parameters()

        optimizer = optim.Adam(params, lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])  #

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.config["scheduler_factor"],
                                                         patience=self.config["scheduler_patience"])
        early_stopping = EarlyStopping(patience=self.config["early_stopping_patience"],
                                       min_delta=self.config["early_stopping_min_delta"])
        engine = TrainingEngine(model, self.config["device"], criterion, optimizer, scheduler, num_classes)

        # --- Wczytywanie wag (Robustness Mode) ---
        if self.config.get("robustness_mode", False):
            path = os.path.join(self.test_dir, "saved_models", f"model_weights_fold_{fold_num}.pth")
            #path = self.config.get("trained_model_path")
            if path and os.path.exists(path):
                model.load_state_dict(torch.load(path))
                print(f"[MODE] Tryb Odporności: Wczytano wagi. Pomijam trening.")
            else:
                raise FileNotFoundError(f"Błąd! robustness_mode=True, ale nie znaleziono pliku: {path}")

        # 4. Pętla treningowa
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        epoch_durations_sec = []
        vram_mb = 0
        model_path = os.path.join(self.config["save_dir"], f"model_weights_fold_{fold_num}.pth")
        epoch_ram_logs = []
        epoch_cpu_logs = []

        if not self.config.get("robustness_mode", False):
            for epoch in range(self.config["num_epochs"]):
                start_ep = time.time()
                t_loss, t_acc = engine.train_one_epoch(train_loader, epoch + 1,
                                                       self.config["num_epochs"],
                                                       fold_num)
                v_loss, v_acc = engine.validate(val_loader)

                ep_time = time.time() - start_ep
                epoch_durations_sec.append(ep_time)

                train_losses.append(t_loss)
                val_losses.append(v_loss)
                train_accs.append(t_acc)
                val_accs.append(v_acc)

                vram_mb = torch.cuda.max_memory_allocated(self.config["device"]) / (1024 ** 2) if "cuda" in \
                                                                                                  self.config[
                                                                                                      "device"] else 0
                process = psutil.Process(os.getpid())
                ram_mb = process.memory_info().rss / (1024 ** 2)  # Zużycie RAM przez proces w MB
                cpu_usage = psutil.cpu_percent()  # Procentowe obciążenie CPU
                epoch_ram_logs.append(ram_mb)
                epoch_cpu_logs.append(cpu_usage)

                logger.log(self.config, {
                    "fold": fold_num,
                    "epoch": epoch + 1,
                    "best_epoch": early_stopping.best_epoch,
                    "train_loss": t_loss,
                    "train_acc": t_acc,
                    "val_loss": v_loss,
                    "val_acc": v_acc,
                    "ram_mb": ram_mb,
                    "cpu_usage": cpu_usage,
                    "vram_mb": vram_mb,
                    "params_count": params_count,
                    "train_time_sec": ep_time
                })

                print(f"Fold {fold_num} Ep {epoch + 1}: Train Loss={t_loss:.4f}, Val Acc={v_acc:.4f}")

                # Zapisujemy wagi tylko gdy model się poprawił (Best Model)
                if self.config.get("use_early_stopping", True):
                    if early_stopping(v_loss, epoch + 1):
                        print(f"[EarlyStopping] Zatrzymano fold {fold_num} na epoce {epoch + 1}.")
                        break

                    if early_stopping.improved:
                        torch.save(model.state_dict(), model_path)
                else:
                    # Jeśli nie ma EarlyStopping, zapisujemy po prostu każdą epokę
                    torch.save(model.state_dict(), model_path)

        else:
            print(f"    [SKIP] Robustness Mode aktywny - pomijam proces uczenia.")
            train_losses, val_losses, train_accs, val_accs = [0], [0], [0], [0]
            epoch_durations_sec = [0]
            if "cuda" in self.config["device"]:
                vram_mb = torch.cuda.max_memory_allocated(self.config["device"]) / (1024 ** 2)

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Wczytano najlepsze wagi z epoki {early_stopping.best_epoch} do testów.")

        # Oddelegowanie logiki ewaluacji do osobnej metody (poprawa czytelności)
        acc = self._evaluate_and_log_fold(
            model, test_loader, criterion, num_classes, class_names, fold_num,
            train_losses, val_losses, train_accs, val_accs, vram_mb, params_count,
            epoch_durations_sec, early_stopping, model_path, start_fold_total, logger,
            epoch_ram_logs, epoch_cpu_logs
        )

        # Twarde czyszczenie pamięci zapobiegające wyciekom VRAM między foldami (OOM prevention)
        del model, optimizer, scheduler, engine, criterion
        if "cuda" in self.config["device"]:
            torch.cuda.empty_cache()

        return acc

    def _evaluate_and_log_fold(self, model, test_loader, criterion, num_classes, class_names, fold_num,
                               train_losses, val_losses, train_accs, val_accs, vram_mb, params_count,
                               epoch_durations_sec, early_stopping, model_path, start_fold_total, logger,
                               epoch_ram_logs, epoch_cpu_logs):
        """
        Zamyka cykl życia folda: generuje wykresy, oblicza metryki testowe,
        wykonuje wiarygodny profil wydajności inferencji i archiwizuje logi.
        """
        # 5. Ewaluacja i Plotting
        curves_save_path = os.path.join(self.plots_dir, f"fold_{fold_num}_curves.png")
        Visualizer.plot_learning_curves(train_losses, val_losses, train_accs, val_accs, fold_num, curves_save_path)

        cm_save_path = os.path.join(self.plots_dir, f"fold_{fold_num}_confusion_matrix.png")
        evaluator = ModelEvaluator(num_classes, self.config["device"], self.config)
        test_metrics = evaluator.evaluate(model, test_loader, criterion, class_names, fold_num, cm_save_path)

        # 6. Dodatkowe miary
        os.makedirs(self.config["save_dir"], exist_ok=True)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        # Inferencja
        model.eval()
        dummy = torch.randn(1, 3, self.config["img_size"], self.config["img_size"]).to(self.config["device"])

        with torch.no_grad():
            # Rozgrzewka (Warm-up) GPU eliminująca błąd narzutu inicjalizacyjnego
            for _ in range(5):
                _ = model(dummy)
            if "cuda" in self.config["device"]:
                torch.cuda.synchronize()  # Bariera synchronizacyjna asynchronicznych operacji CUDA

            start_inf = time.time()
            for _ in range(10):
                _ = model(dummy)
            if "cuda" in self.config["device"]:
                torch.cuda.synchronize()

        inf_ms = ((time.time() - start_inf) / 10) * 1000

        # --- OPCJONALNY GRAD-CAM ---
        if self.config.get("use_gradcam", False):
            self._handle_explainability(model, test_metrics, fold_num, class_names)

        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 ** 2)
        cpu_usage = psutil.cpu_percent()

        # 7. Summary log
        logger.log(self.config, {
            "fold": fold_num, "best_epoch": early_stopping.best_epoch,
            "test_acc": test_metrics['acc'], "test_prec": test_metrics['prec'],
            "test_rec": test_metrics['rec'], "test_f1": test_metrics['f1'],
            "ram_mb": max(epoch_ram_logs) if epoch_ram_logs else 0,   # Peak
            "cpu_usage": np.mean(epoch_cpu_logs) if epoch_cpu_logs else 0, # Average
            "vram_mb": vram_mb, "params_count": params_count, "model_size_mb": model_size_mb,
            "inference_ms": inf_ms, "train_time_sec": sum(epoch_durations_sec),
            "total_fold_time_sec": time.time() - start_fold_total
        }, row_type="SUMMARY")

        return test_metrics['acc']

    def _handle_explainability(self, model, metrics, fold_num, class_names):
        """
        Zarządza modułem interpretowalności wizualnej (Explainable AI).
        Generuje mapy aktywacji Grad-CAM dla najlepiej i najgorzej sklasyfikowanych próbek.
        """
        # Leniwy import zapobiegający obciążaniu systemu, gdy Grad-CAM jest wyłączony
        from evaluation.gradcam import GradCAM, CAMVisualizer
        from PIL import Image
        from data.transforms import TransformFactory

        # Zapis do TXT (Append) - wykonuje się przed generowaniem obrazów
        log_gradcam_paths(self.config["gradcam_txt_path"], fold_num, metrics['top_successes'], metrics['top_failures'])

        # Generowanie obrazów
        os.makedirs(self.gradcam_img_dir, exist_ok=True)
        target_layer = model.model.features[-1]
        gcam = GradCAM(model, target_layer)
        test_transforms = TransformFactory.get_test_transforms(self.config)

        # Wybieramy sukcesy i błędy do rysowania
        to_plot = [*(metrics['top_successes'][:self.config.get("num_visualizations", 3)]),
                   *(metrics['top_failures'][:self.config.get("num_visualizations", 3)])]

        for s in to_plot:
            img_pil = Image.open(s['path']).convert("RGB")
            tensor = test_transforms(img_pil).unsqueeze(0).to(self.config["device"])

            # NAPRAWA BŁĘDU MEAN() W GRAD-CAM:
            # Wymuszamy śledzenie gradientów dla tego konkretnego obrazu.
            # Zmusza to zamrożoną sieć do policzenia gradientów wstecz, co aktywuje mechanizmy XAI.
            tensor.requires_grad_(True)

            heatmap = gcam.generate_heatmap(tensor)

            tag = "SUCCESS" if s['correct'] else "FAIL"
            fname = f"F{fold_num}_{tag}_{int(s['confidence'] * 100)}_{os.path.basename(s['path'])}"
            CAMVisualizer.save_visualization(s['path'], heatmap, os.path.join(self.gradcam_img_dir, fname))

    def _finalize_experiment(self, start_time, scores, class_names):
        """
        Metoda zamykająca globalny proces eksperymentu.
        Generuje ostateczne statystyki czasowe i skutecznościowe oraz zapisuje mapę klas.
        """
        total_sec = time.time() - start_time
        print(f"\n{'#' * 40}\nKONIEC EKSPERYMENTU")
        print(f"Całkowity czas: {total_sec / 60:.2f} minut")

        # POPRAWKA: cv_scores to teraz lista floatów, np.mean zadziała bezbłędnie
        print(f"ŚREDNIA SKUTECZNOŚĆ: {np.mean(scores) * 100:.2f}% ± {np.std(scores) * 100:.2f}%\n{'#' * 40}")

        # Zapisywanie model_info.json niezbędnego do późniejszej inferencji (Predictor)
        model_info = {
            "model_name": self.config["model_name"],
            "head_depth": self.config.get("head_depth", 1),
            "head_width": self.config.get("head_width", 512),
            "img_size": self.config["img_size"],
            "norm_mean": self.config["norm_mean"],
            "norm_std": self.config["norm_std"]
        }
        with open(os.path.join(self.config["save_dir"], "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=4)

        with open(os.path.join(self.config["save_dir"], "class_names.json"), "w") as f:
            json.dump(class_names, f)