import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image
import json
import os
import time  # Dodajemy do mierzenia czasu

# --- KONFIGURACJA ---
IMAGE_PATH = "testZdjecieA10.jpg"
MODEL_WEIGHTS = "model_weights.pth"
CLASS_NAMES_JSON = "class_names.json"


# --- KLASY (MUSZĄ BYĆ IDENTYCZNE JAK W TRENINGU) ---

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(image, padding, 0, 'constant')


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# --- KLASA WRAPPER (ŁADUJE MODEL RAZ I TRZYMA GO W PAMIĘCI) ---

class Predictor:
    def __init__(self, model_path, json_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ładowanie nazw klas
        with open(json_path, "r") as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)

        # Inicjalizacja modelu
        self.model = EfficientNetClassifier(num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Definicja transformacji
        self.transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"Model zainicjalizowany na urządzeniu: {self.device}")

    def predict(self, img_path):
        # 1. Start pomiaru czasu
        start_time = time.time()

        # 2. Przygotowanie zdjęcia
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # 3. Sama predykcja
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]

        # 4. Koniec pomiaru czasu
        end_time = time.time()
        inference_time = end_time - start_time  # Wynik w sekundach

        # Wyciągnięcie wyników
        prob_values, indices = torch.sort(probabilities, descending=True)

        return prob_values, indices, inference_time


# --- SYMULACJA DZIAŁANIA W GUI ---

if __name__ == "__main__":
    # KROK 1: Start GUI / Inicjalizacja (tylko raz!)
    print("Ładowanie modelu do pamięci... proszę czekać.")
    try:
        engine = Predictor(MODEL_WEIGHTS, CLASS_NAMES_JSON)
    except FileNotFoundError:
        print("Błąd: Nie znaleziono plików modelu. Uruchom najpierw trening.")
        exit()

    # KROK 2: Użytkownik klika "Sprawdź zdjęcie" (można to powtarzać w pętli)
    #while True:
    path = IMAGE_PATH#input("\nPodaj ścieżkę do zdjęcia (lub 'q' aby wyjść): ")
    #if path.lower() == 'q':
    #    break

    if not os.path.exists(path):
        print("Błąd: Plik nie istnieje!")
    #    continue

    # WYWOŁANIE PREDYKCJI
    probs, idxs, t = engine.predict(path)

    # WYŚWIETLENIE WYNIKÓW
    print(f"Czas analizy: {t:.4f} sekundy ({t * 1000:.2f} ms)")
    print("-" * 30)
    for i in range(0, engine.num_classes):
        p = probs[i].item() * 100
        name = engine.class_names[idxs[i]]
        print(f"{name:20} : {p:.2f}%")