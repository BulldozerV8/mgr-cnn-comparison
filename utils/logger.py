import csv
import os


class CSVLogger:
    """
    Klasa odpowiedzialna za systematyczne rejestrowanie wyników eksperymentów w formacie CSV.
    Umożliwia śledzenie zarówno parametrów konfiguracyjnych, jak i metryk wydajnościowych
    oraz sprzętowych modelu.
    """

    def __init__(self, filename):
        """
        Inicjalizuje logger i przygotowuje plik wynikowy.

        Args:
            filename (str): Ścieżka do pliku CSV, w którym zapisywane będą logi.
        """
        self.filename = filename

        # Definicja nagłówków kolumn - zapewnia spójną strukturę danych w pliku
        self.headers = [
            "type", "test_name", "dataset_tag", "model_name", "pretrained", "unfreeze_blocks",
            "batch_size", "learning_rate", "label_smoothing", "head_depth", "head_width", "head_dropout",
            "aug_mode", "weather_prob", "aspect_ratio_mode", "num_samples",
            "use_cv", "fold", "epoch", "best_epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "test_acc", "test_prec", "test_rec", "test_f1", "ram_mb", "cpu_usage",
            "vram_mb", "params_count", "trainable_params", "model_size_mb", "inference_ms",
            "train_time_sec", "total_fold_time_sec"
        ]

        # Weryfikacja istnienia folderu docelowego i jego ewentualne utworzenie
        if os.path.dirname(self.filename):
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        # Otwarcie pliku w trybie dopisywania ('a'). Za każdym razem, gdy obiekt jest
        # inicjalizowany, dopisywany jest nowy zestaw nagłówków. Tworzy to czytelny
        # podział na sekcje (tabele) wewnątrz jednego pliku zbiorczego.
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log(self, config, metrics, row_type="EPOCH"):
        """
        Zapisuje pojedynczy wiersz danych (epokę lub podsumowanie testu) do pliku.

        Args:
            config (dict): Słownik z globalną konfiguracją eksperymentu.
            metrics (dict): Słownik zawierający bieżące metryki (loss, acc, czasy, VRAM).
            row_type (str): Etykieta typu wiersza (np. "EPOCH", "FINAL_FOLD", "TEST").
        """

        # Budowa wiersza z wykorzystaniem metody .get() dla maksymalnego bezpieczeństwa.
        # Zapobiega to przerwaniu długotrwałego treningu w przypadku braku pojedynczego
        # klucza w słowniku metryk lub konfiguracji.
        row = [
            row_type,
            config.get("test_name", "-"),
            config.get("dataset_tag", "-"),
            config.get("model_name", "-"),
            config.get("pretrained", "-"),
            config.get("unfreeze_blocks", 0),
            config.get("batch_size", "-"),
            config.get("learning_rate", "-"),
            config.get("label_smoothing", "-"),
            config.get("head_depth", 1),
            config.get("head_width", "-"),
            config.get("head_dropout", "-"),
            config.get("aug_mode", "-"),
            config.get("weather_prob", 0),
            config.get("aspect_ratio_mode", "Resize"),
            config.get("num_samples_per_epoch", "-"),
            config.get("use_cv", False),
            metrics.get("fold", "-"),
            metrics.get("epoch", "-"),
            metrics.get("best_epoch", "-"),

            # Formatowanie wartości zmiennoprzecinkowych dla zachowania czystości danych
            f"{metrics.get('train_loss', 0):.4f}",
            f"{metrics.get('train_acc', 0):.4f}",
            f"{metrics.get('val_loss', 0):.4f}",
            f"{metrics.get('val_acc', 0):.4f}",
            f"{metrics.get('test_acc', 0):.4f}",
            f"{metrics.get('test_prec', 0):.4f}",
            f"{metrics.get('test_rec', 0):.4f}",
            f"{metrics.get('test_f1', 0):.4f}",

            # Monitorowanie zasobów sprzętowych i wydajności czasowej
            f"{metrics.get('ram_mb', 0):.2f}",
            f"{metrics.get('cpu_usage', 0):.2f}",
            f"{metrics.get('vram_mb', 0):.2f}",
            metrics.get("params_count", 0),
            metrics.get("trainable_params", 0),
            f"{metrics.get('model_size_mb', 0):.2f}",
            f"{metrics.get('inference_ms', 0):.4f}",
            f"{metrics.get('train_time_sec', 0):.2f}",
            f"{metrics.get('total_fold_time_sec', 0):.2f}"
        ]

        # Fizyczny zapis danych na dysk w trybie append
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)