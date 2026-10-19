import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SquarePad:
    """
    Transformacja wyrównująca proporcje obrazu do kwadratu poprzez dodanie czarnych pasów (padding).
    Zapobiega deformacji obiektów (np. spłaszczeniu samolotów) podczas zmiany rozmiaru.
    """

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(image, padding, 0, 'constant')


class WeatherTransformWrapper:
    """
    Zaawansowany wrapper dla biblioteki Albumentations, realizujący efekty pogodowe.
    Wykorzystuje szybkie operacje na macierzach NumPy do symulacji deszczu, śniegu i mgły.
    """

    def __init__(self, config):
        # Sprawdzamy tryb proporcji, aby zachować spójność z transformacjami standardowymi
        aspect_mode = config.get("aspect_ratio_mode", "SquarePad")

        transforms_list = []

        # Logika dopasowania rozmiaru przed nałożeniem pogody
        if aspect_mode == "Crop":
            transforms_list.append(A.RandomResizedCrop(
                size=(config["img_size"], config["img_size"]),
                scale=config.get("aug_scale", (0.8, 1.0))
            ))
        else:
            # Dla SquarePad i Stretch najpierw skalujemy, zachowując logikę z configu
            transforms_list.append(A.Resize(config["img_size"], config["img_size"]))

        # Nakładanie efektów pogodowych
        transforms_list.extend([
            A.OneOf([
                A.RandomRain(p=1.0),
                A.RandomSnow(p=1.0),
                A.RandomFog(p=1.0),
            ], p=config["weather_prob"]),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=config["norm_mean"], std=config["norm_std"]),
            ToTensorV2(),
        ])

        self.alb_transform = A.Compose(transforms_list)

    def __call__(self, image):
        img_np = np.array(image)
        augmented = self.alb_transform(image=img_np)
        return augmented['image']


class TransformFactory:
    """
    Fabryka transformacji dostarczająca zestawy augmentacji dla fazy treningu i testów.
    Obsługuje automatyczne wzmacnianie danych (TrivialAugmentWide) oraz testy odporności.
    """

    @staticmethod
    def get_train_transforms(config):
        """
        Generuje potok transformacji treningowych z uwzględnieniem wybranego trybu augmentacji.
        """
        if config["aug_mode"] == "standard":
            aspect_ratio_mode = config.get("aspect_ratio_mode", "SquarePad")
            t_list = []

            # 1. Obsługa proporcji obrazu
            if aspect_ratio_mode == "SquarePad":
                t_list.append(SquarePad())
                t_list.append(transforms.Resize((config["img_size"], config["img_size"])))
            elif aspect_ratio_mode == "Stretch":
                t_list.append(transforms.Resize((config["img_size"], config["img_size"])))
            elif aspect_ratio_mode == "Crop":
                t_list.append(transforms.RandomResizedCrop(
                    config["img_size"],
                    scale=config.get("aug_scale", (0.8, 1.0))
                ))

            # 2. Główne augmentacje
            t_list.extend([
                transforms.TrivialAugmentWide(),  # Automatyczny dobór parametrów transformacji
                transforms.RandomHorizontalFlip(),  # Lustrzane odbicie poziome
                transforms.ToTensor(),  # Konwersja do Tensora (0.0 - 1.0)
                transforms.Normalize(mean=config["norm_mean"], std=config["norm_std"]),
                transforms.RandomErasing(
                    p=config.get("erasing_p", 0.5),
                    scale=(0.02, 0.2)
                )
            ])
            return transforms.Compose(t_list)

        elif config["aug_mode"] == "weather":
            return WeatherTransformWrapper(config)

    @staticmethod
    def get_test_transforms(config):
        """
        Generuje minimalny potok transformacji dla fazy walidacji i testów.
        Zapewnia powtarzalność wyników poprzez brak operacji losowych.
        """
        # Jeśli badamy odporność na pogodę, używamy WeatherTransformWrapper
        if config.get("robustness_mode", False) and config.get("aug_mode") == "weather":
            return WeatherTransformWrapper(config)

        # Standardowy, czysty potok testowy
        t_list = []
        if config.get("aspect_ratio_mode") == "SquarePad":
            t_list.append(SquarePad())

        t_list.extend([
            transforms.Resize((config["img_size"], config["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["norm_mean"], std=config["norm_std"])
        ])
        return transforms.Compose(t_list)