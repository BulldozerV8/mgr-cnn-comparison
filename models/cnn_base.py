from torch import nn
import torchvision.models as models
from models.custom_model import MyCustomCNN

class CNN(nn.Module):
    """
    Główna klasa modelu sieci neuronowej.
    Obsługuje różne architektury (EfficientNet, ResNet) oraz umożliwia zaawansowany
    Transfer Learning poprzez selektywne odmrażanie bloków bazy i dynamiczną budowę głowicy.
    """

    # ZMIENIONO: Rozszerzono sygnaturę __init__ o nowe parametry, aby klasa główna mogła je przechwycić z CONFIG
    def __init__(self, num_classes, model_name="efficientnet_b0", pretrained=True,
                 head_depth=1, head_width=512, head_dropout=0.3, unfreeze_blocks=0, pooling_mode="avg",
                 differential_lr=False, custom_base_channels=32, custom_block_repeats=[2, 2, 2],
                 custom_activation="silu",
                 use_se_block=True, se_reduction=16, use_depthwise=True, drop_path_rate=0.2,
                 expansion_ratio=4, kernel_size=3, **kwargs):
        """
        Inicjalizacja modelu z parametrami badawczymi.

        Args:
            num_classes (int): Liczba klas wyjściowych (np. typy samolotów).
            model_name (str): Nazwa architektury bazowej.
            pretrained (bool): Czy użyć wag wyuczonych na ImageNet.
            head_depth (int): Liczba warstw liniowych w głowicy klasyfikującej.
            head_width (int): Liczba neuronów w warstwach ukrytych głowicy.
            head_dropout (float): Współczynnik regularyzacji Dropout.
            unfreeze_blocks (int): Liczba końcowych bloków bazy do odmrożenia (Fine-tuning).
        """
        super(CNN, self).__init__()

        if model_name == "efficientnet_b0":
            # 1. Ładowanie modelu i wag bazowych
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)

            # 1a. Konfiguracja warstwy Poolingu (Global Pooling)
            # Podmiana domyślnego Average Pooling na Max Pooling, jeśli wybrano w configu
            if pooling_mode == "max":
                self.model.avgpool = nn.AdaptiveMaxPool2d(1)  #
            else:
                self.model.avgpool = nn.AdaptiveAvgPool2d(1)  # Domyślny

            # 2. Mechanizm mrożenia/odmrażania (Fine-tuning)
            # Domyślnie mrozimy wszystkie parametry bazy, aby nie utracić wyuczonych cech
            for param in self.model.parameters():
                param.requires_grad = False

            # Selektywne odmrażanie końcowych bloków (features) dla lepszej adaptacji do danych lotniczych
            # EfficientNet-B0 posiada bloki o indeksach od 0 do 8.
            if unfreeze_blocks > 0:
                for i in range(unfreeze_blocks):
                    idx = 8 - i  # Odliczamy od końca
                    if idx >= 0:
                        for param in self.model.features[idx].parameters():
                            param.requires_grad = True

            # 3. Dynamiczna głowica (Head Upgrades)
            # Pobieramy liczbę cech wejściowych z oryginalnego klasyfikatora
            in_features = self.model.classifier[1].in_features
            layers = []
            curr_in = in_features

            # Budowa warstw pośrednich głowicy (jeśli head_depth > 1)
            for _ in range(head_depth - 1):
                layers.append(nn.Linear(curr_in, head_width))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(head_dropout))
                curr_in = head_width

            # Finalna warstwa klasyfikująca dopasowana do liczby kategorii
            layers.append(nn.Linear(curr_in, num_classes))

            # Podmiana oryginalnego klasyfikatora na nową strukturę Sequential
            self.model.classifier[1] = nn.Sequential(*layers)

            # Upewnienie się, że nowa głowica jest zawsze trenowalna
            for param in self.model.classifier[1].parameters():
                param.requires_grad = True

            # 4. Przygotowanie pod wyjaśnialność (Grad-CAM)
            # Referencja do ostatniej warstwy splotowej dla generowania map ciepła
            self.last_conv_layer = self.model.features[-1]

        elif model_name == "my_custom_cnn":
            self.model = MyCustomCNN( in_channels=3, num_classes=num_classes, head_dropout=head_dropout, head_width=head_width,
                                      head_depth=head_depth, pooling_mode=pooling_mode, custom_base_channels=custom_base_channels,
                                      custom_block_repeats=custom_block_repeats, custom_activation=custom_activation, use_se_block=use_se_block,
                                      se_reduction=se_reduction, use_depthwise=use_depthwise, drop_path_rate=drop_path_rate,
                                      expansion_ratio=expansion_ratio, kernel_size=kernel_size
                                    )
            self.last_conv_layer = self.model.last_conv_layer

        elif model_name == "resnet18":
            # Alternatywna architektura bazowa - ResNet18
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            # Standardowa podmiana ostatniej warstwy w pełni połączonej
            self.model.fc = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Nieznana architektura modelu: {model_name}")

    def forward(self, x):
        """
        Definiuje przepływ danych przez sieć (Forward Pass).
        """
        return self.model(x)

    def get_model_stats(self):
        """
        Oblicza statystyki parametrów modelu na potrzeby logowania danych sprzętowych.

        Returns:
            dict: Słownik zawierający całkowitą liczbę parametrów oraz liczbę wag trenowalnych.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params
        }

    def get_last_conv_layer(self):
        """
        Udostępnia ostatnią warstwę splotową dla algorytmu Grad-CAM.
        Dzięki temu mechanizmowi moduł wyjaśnialności jest niezależny od nazewnictwa wewnątrz modelu.
        """
        if hasattr(self, 'last_conv_layer'):
            return self.last_conv_layer
        return None