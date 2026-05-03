import torch

CONFIG = {
        # === PODSTAWOWA KONFIGURACJA TESTU ===
        "test_name": "Test_dzialania_po_upiekszaniu_1",  # Nazwa folderu z wynikami w folderze results
        "dataset_tag": "Zbliska_20",  # Etykieta identyfikująca zbiór danych w raportach
        "aspect_ratio_mode": "SquarePad",  # Metoda dopasowania proporcji: "SquarePad", "Stretch" lub "Crop"

        # --- Sekcja modelu ---
        "unfreeze_blocks": 1,        # 0 = tylko głowica, 1 = ostatni blok + głowica, itd.
        "head_dropout": 0.3,         # Intensywność Dropoutu w głowicy (0.1 - 0.5)
        "head_depth": 1,  # Głębokość głowicy klasyfikującej: 1 = standard, 2-3 = dodatkowe warstwy FC
        "head_width": 512,  # Liczba neuronów w dodatkowych warstwach głowicy

        # --- DANE I PODZIAŁY ---
        "data_path": "datasets/Dataset-small/",
        "results_path": "results", # Folder bazowy dla wszystkich eksperymentów
        "save_dir": "results",    # Lokalizacja zapisu wag (Runner i tak stworzy podfolder testu)
        "aug_mode": "standard", # Tryb augmentacji: "standard" lub "weather" (efekty pogodowe)
        "weather_prob": 0.5,    # Prawdopodobieństwo wystąpienia efektu pogodowego (dla trybu weather)
        "test_size": 0.15,      # Odsetek danych wydzielony jako stały zbiór testowy (hold-out)
        "val_size": 0.20,       # Odsetek danych do walidacji (używane tylko gdy use_cv = False)
        "use_cv": False,        # Przełącznik: True = 5-fold Cross-Validation, False = pojedynczy trening
        "num_folds": 5,         # Liczba podziałów (foldów) dla walidacji krzyżowej
        "use_seed": False,      # Czy wymusić stałe ziarno losowości dla powtarzalności wyników
        "seed": 50,             # Wartość ziarna losowości

        # --- PARAMETRY TECHNICZNE ---
        "img_size": 224,        # Rozdzielczość wejściowa obrazu (EfficientNet-B0 standardowo 224x224)
        "batch_size": 64,       # Liczba próbek przetwarzanych naraz
        "num_workers": 4,       # Liczba procesów procesora do ładowania danych (bezpieczna wartość dla Windows)
        "model_name": "efficientnet_b0",
        "pretrained": True,     # Transfer Learning: True = wczytaj wagi wyuczone na zbiorze ImageNet

        # --- PARAMETRY UCZENIA (HIPERPARAMETRY) ---
        "num_epochs": 3,        # Maksymalna liczba epok (pełnych przejść przez zbiór danych)
        "num_samples_per_epoch": 30000,  # Liczba zdjęć na epokę (wymuszone przez Oversampling)
        "learning_rate": 0.0001,
        "weight_decay": 1e-4,   # Siła regularyzacji L2 (zapobiega zbyt dużym wagom modelu)
        "label_smoothing": 0.1, # Wygładzanie etykiet (poprawia generalizację, zapobiega pewności siebie modelu)

        # --- EARLY STOPPING I SCHEDULER ---
        "early_stopping_patience": 3,   # Ile epok czekać na poprawę Val Loss przed przerwaniem
        "early_stopping_min_delta": 0.01, # Minimalna zmiana Val Loss uznawana za poprawę
        "use_early_stopping": False,    # Czy automatycznie przerywać naukę przy braku postępów
        "scheduler_patience": 1,        # Ile epok czekać przed redukcją Learning Rate
        "scheduler_factor": 0.5,        # Współczynnik redukcji Learning Rate (np. z 0.0001 na 0.00005)

        # --- GRAD-CAM I ANALIZA WYJAŚNIALNOŚCI ---
        "use_gradcam": True,              # Czy generować mapy ciepła (Visual Explanations)
        "num_visualizations": 5,          # Liczba najlepiej i najgorzej ocenionych zdjęć do analizy na fold
        "metrics_average": "macro",       # Typ uśredniania metryk (macro, weighted, micro)

        # --- TESTY ODPORNOŚCI (ROBUSTNESS TESTING) ---
        "robustness_mode": False,  # True = wczytuje gotowy model z trained_model_path i tylko testuje
        "trained_model_path": "results/EffNet_Baseline_V1/saved_models/model_weights_fold_1.pth",

        # --- PARAMETRY PRZETWARZANIA I ŚRODOWISKO ---
        "norm_mean": [0.485, 0.456, 0.406], # Średnia dla normalizacji ImageNet
        "norm_std": [0.229, 0.224, 0.225],  # Odchylenie std dla normalizacji ImageNet
        "aug_scale": (0.8, 1.0),            # Zakres losowego skalowania obrazu podczas treningu
        "erasing_p": 0.5,                    # Prawdopodobieństwo losowego usuwania fragmentu obrazu (Random Erasing)
        "device": "cuda" if torch.cuda.is_available() else "cpu" # Automatyczny wybór GPU lub CPU
    }