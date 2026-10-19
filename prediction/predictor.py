import torch
import torch.nn.functional as F
import os
import json
import time
import csv
from datetime import datetime
from PIL import Image
from data.transforms import TransformFactory
from models.cnn_base import CNN


class Predictor:
    """
    Zaawansowany silnik inferencyjny (Inference Engine).
    Samodzielnie konfiguruje architekturę na podstawie metadanych modelu,
    zarządza procesem predykcji i automatycznie archiwizuje wyniki.
    """

    def __init__(self, model_weights_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 1. Automatyczne wykrywanie ścieżek i ładowanie metadanych
        model_dir = os.path.dirname(model_weights_path)
        info_path = os.path.join(model_dir, "model_info.json")
        classes_path = os.path.join(model_dir, "class_names.json")

        if not os.path.exists(info_path) or not os.path.exists(classes_path):
            raise FileNotFoundError("Błąd: W folderze modelu brakuje plików model_info.json lub class_names.json!")

        with open(info_path, "r") as f:
            self.model_info = json.load(f)
        with open(classes_path, "r") as f:
            self.class_names = json.load(f)

        # 2. Dynamiczna inicjalizacja architektury i transformacji
        self.transform = TransformFactory.get_test_transforms(self.model_info)
        self.model = CNN(
            num_classes=len(self.class_names),
            model_name=self.model_info["model_name"],
            head_depth=self.model_info.get("head_depth", 3),
            head_width=self.model_info.get("head_width", 256),
            pooling_mode=self.model_info.get("pooling_mode", "avg"),
            custom_base_channels=self.model_info.get("custom_base_channels", 32),
            custom_block_repeats=self.model_info.get("custom_block_repeats", [3, 3, 4]),
            use_se_block=self.model_info.get("use_se_block", True),
            use_depthwise=self.model_info.get("use_depthwise", True),
            kernel_size=self.model_info.get("kernel_size", 5),
            expansion_ratio=self.model_info.get("expansion_ratio", 4)
        )

        # Bezpieczne ładowanie wag
        state_dict = torch.load(model_weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 3. Przygotowanie systemu logowania wyników (Session Tracking)
        self._setup_session_logging(model_dir)

    def _setup_session_logging(self, model_dir):
        """Tworzy strukturę folderów dla sesji predykcji i inicjalizuje plik CSV."""
        # Wyjście o 2 poziomy w górę (z saved_models do folderu testu)
        test_dir = os.path.abspath(os.path.join(model_dir, ".."))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.session_dir = os.path.join(test_dir, "inference_results", f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)

        self.log_path = os.path.join(self.session_dir, "predictions_log.csv")

        # Inicjalizacja nagłówka CSV
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Image_Name", "Top1_Class", "Confidence", "TopK_Results", "Inference_ms"])

    @torch.inference_mode()
    def predict(self, image_path, top_k=3):
        """
        Wykonuje predykcję dla pojedynczego obrazu.
        Zwraca ustrukturyzowany wynik i zapisuje go w logach.
        """
        start_time = time.time()
        try:
            # Przetwarzanie obrazu
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Inferencja
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

            # Wyciąganie wyników Top-K
            topk_prob, topk_idx = torch.topk(probabilities, top_k)

            inf_time_ms = (time.time() - start_time) * 1000

            results = []
            for i in range(top_k):
                results.append({
                    "label": self.class_names[topk_idx[i]],
                    "confidence": topk_prob[i].item()
                })

            # Logowanie do CSV
            self._log_to_csv(image_path, results, inf_time_ms)

            return {
                "status": "success",
                "image": os.path.basename(image_path),
                "top_k": results,
                "time_ms": inf_time_ms
            }

        except Exception as e:
            return {"status": "error", "message": str(e), "image": os.path.basename(image_path)}

    def _log_to_csv(self, image_path, results, inf_time_ms):
        """Pomocniczy zapis pojedynczej linii do pliku sesji."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        top1_label = results[0]["label"]
        top1_conf = f"{results[0]['confidence']:.4f}"

        # Top-K w formacie: Klasa(Prob); Klasa(Prob)
        topk_str = " | ".join([f"{r['label']}({r['confidence']:.2f})" for r in results])

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [timestamp, os.path.basename(image_path), top1_label, top1_conf, topk_str, f"{inf_time_ms:.2f}"])

    def predict_folder(self, folder_path, top_k=3):
        """Przetwarza masowo wszystkie obrazy w folderze."""
        supported_ext = (".jpg", ".jpeg", ".png", ".bmp")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_ext)]

        print(f"Rozpoczynam masowe przetwarzanie: {len(files)} plików.")
        all_results = []

        for file_name in files:
            full_path = os.path.join(folder_path, file_name)
            res = self.predict(full_path, top_k=top_k)
            all_results.append(res)

        print(f"Zakończono. Wyniki zapisano w: {self.session_dir}")
        return all_results