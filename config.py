import torch

CONFIG = {
        # === PODSTAWOWA KONFIGURACJA TESTU ===
        "test_name": "Test",  # Nazwa folderu z wynikami w folderze results
        "dataset_tag": "Aircraft_close",  # Etykieta identyfikująca zbiór danych w raportach
        "aspect_ratio_mode": "SquarePad",  # Metoda dopasowania proporcji: "SquarePad", "Stretch" lub "Crop"

        # --- Konfiguracja Architektury Modelu ---
        "model_name": "my_custom_cnn", #efficientnet_b0, my_custom_cnn
        "pretrained": True,     # Transfer Learning: True = wczytaj wagi wyuczone na zbiorze ImageNet
        "img_size": 448,        # Rozdzielczość wejściowa obrazu (EfficientNet-B0 standardowo 224x224)

        # --- Strategia Fine-tuningu ---
        "unfreeze_blocks": 4,        # 0 = tylko głowica, 1 = ostatni blok + głowica, itd.
        "head_dropout": 0.3,         # Intensywność Dropoutu w głowicy (0.1 - 0.5)
        "head_depth": 3,  # Głębokość głowicy klasyfikującej: 1 = standard, 2-3 = dodatkowe warstwy FC
        "head_width": 256,  # Liczba neuronów w dodatkowych warstwach głowicy
        "pooling_mode": "avg",       # Tryb agregacji cech: "avg" (średnia) lub "max" (maksimum), NIE DZIAŁA gem
        "differential_lr": False,    # True = mniejszy LR dla bazy, większy dla głowicy

        # --- DANE I PODZIAŁY ---
        "data_path": "datasets/Aircraft_close/",
        "results_path": "results", # Folder bazowy dla wszystkich eksperymentów
        "save_dir": "results",    # Lokalizacja zapisu wag (Runner i tak stworzy podfolder testu)
        "aug_mode": "standard", # Tryb augmentacji: "standard" lub "weather" (efekty pogodowe)
        "weather_prob": 1.0,    # Prawdopodobieństwo wystąpienia efektu pogodowego (dla trybu weather)
        "test_size": 0.15,      # Odsetek danych wydzielony jako stały zbiór testowy (hold-out)
        "val_size": 0.20,       # Odsetek danych do walidacji (używane tylko gdy use_cv = False)
        "use_cv": False,        # Przełącznik: True = 5-fold Cross-Validation, False = pojedynczy trening
        "num_folds": 5,         # Liczba podziałów (foldów) dla walidacji krzyżowej
        "use_seed": True,      # Czy wymusić stałe ziarno losowości dla powtarzalności wyników
        "seed": 50,             # Wartość ziarna losowości

        # --- PARAMETRY TECHNICZNE ---
        "batch_size": 16,       # Liczba próbek przetwarzanych naraz
        "num_workers": 4,       # Liczba procesów procesora do ładowania danych (bezpieczna wartość dla Windows)


        # --- PARAMETRY UCZENIA (HIPERPARAMETRY) ---
        "num_epochs": 30,        # Maksymalna liczba epok (pełnych przejść przez zbiór danych)
        "num_samples_per_epoch": 30000,  # Liczba zdjęć na epokę (wymuszone przez Oversampling)
        "samples_multiplier": 1, #max(30000, N * 1)
        "learning_rate": 0.0001,
        "weight_decay": 1e-04,   # Siła regularyzacji L2 (zapobiega zbyt dużym wagom modelu)
        "label_smoothing": 0.1, # Wygładzanie etykiet (poprawia generalizację, zapobiega pewności siebie modelu)

        # --- EARLY STOPPING I SCHEDULER ---
        "early_stopping_patience": 5,   # Ile epok czekać na poprawę Val Loss przed przerwaniem
        "early_stopping_min_delta": 0.0001, # Minimalna zmiana Val Loss uznawana za poprawę
        "use_early_stopping": True,    # Czy automatycznie przerywać naukę przy braku postępów
        "scheduler_patience": 3,        # Ile epok czekać przed redukcją Learning Rate
        "scheduler_factor": 0.5,        # Współczynnik redukcji Learning Rate (np. z 0.0001 na 0.00005)

        # --- GRAD-CAM I ANALIZA WYJAŚNIALNOŚCI ---
        "use_gradcam": True,              # Czy generować mapy ciepła (Visual Explanations)
        "num_visualizations": 10,          # Liczba najlepiej i najgorzej ocenionych zdjęć do analizy na fold
        "metrics_average": "macro",       # Typ uśredniania metryk (macro, weighted, micro)

        # --- TESTY ODPORNOŚCI (ROBUSTNESS TESTING) ---
        "robustness_mode": False,  # True = wczytuje gotowy model z trained_model_path i tylko testuje
        "trained_model_path": "results/Custom_Final_Ptaki_pogoda/saved_models/model_weights_fold_1.pth",

        # --- PARAMETRY PRZETWARZANIA I ŚRODOWISKO ---
        "norm_mean": [0.485, 0.456, 0.406], # Średnia dla normalizacji ImageNet
        "norm_std": [0.229, 0.224, 0.225],  # Odchylenie std dla normalizacji ImageNet
        "aug_scale": (0.8, 1.0),            # Zakres losowego skalowania obrazu podczas treningu
        "erasing_p": 0.5,                    # Prawdopodobieństwo losowego usuwania fragmentu obrazu (Random Erasing)
        "device": "cuda" if torch.cuda.is_available() else "cpu", # Automatyczny wybór GPU lub CPU
    
        # === AUTORSKA ARCHITEKTURA (Dla model_name: "my_custom_cnn") ===
        "custom_base_channels": 32,          # Początkowa liczba filtrów w pierwszej warstwie splotowej
        "custom_block_repeats": [3, 3, 4],     # Liczba bloków rezydualnych w kolejnych sekcjach sieci (głębokość)
        "custom_activation": "silu",         # Nowoczesna funkcja aktywacji: "silu", "gelu" lub "relu"
        "use_se_block": True,                # Czy użyć bloku uwagi kanałowej Squeeze-and-Excitation (True/False)
        "use_depthwise": True,               # True = konwolucje separowalne (EfficientNet), False = zwykłe (ResNet)
        "drop_path_rate": 0.2,               # Współczynnik regularyzacji Stochastic Depth (DropPath)
        "expansion_ratio": 4,               # Mnożnik rozszerzenia kanałów wewnątrz bloku (Inverted Bottleneck)
        "kernel_size": 5,                   # Rozmiar okna splotu (np. 3 dla 3x3 lub 5 dla 5x5)
        "se_reduction": 16,                  # Raczej do pominięcia przy dostrajaniu # Stopień kompresji szerokości w bloku SE (np. 4, 8, 16)
    }