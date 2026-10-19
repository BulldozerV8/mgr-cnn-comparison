import torch
from torch import nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Moduł Squeeze-and-Excitation (SE).
    Realizuje mechanizm uwagi kanałowej (Channel Attention), pozwalając sieci
    na dynamiczne nadawanie wag poszczególnym kanałom mapy cech. Wspiera model
    w skupianiu się na istotnych detalach obrazu przy jednoczesnym ignorowaniu szumu.
    """

    def __init__(self, in_channels, reduction=16, activation=nn.SiLU()):
        """
        Inicjalizacja bloku uwagi SE.

        Args:
            in_channels (int): Liczba kanałów wejściowych.
            reduction (int): Współczynnik redukcji wymiaru kanałów (kompresja bottleneck).
            activation (nn.Module): Instancja funkcji aktywacji przekazywana z modelu głównego.
        """
        super(SEBlock, self).__init__()

        # 1. Zautomatyzowane wyliczanie kanałów ukrytych (bottleneck) z zabezpieczeniem przed wyzerowaniem
        reduced_channels = max(1, in_channels // reduction)

        # 2. Budowa zwięzłego bloku Squeeze (Pooling) i Excitation (Sploty 1x1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze: redukcja przestrzenna (B, C, H, W) -> (B, C, 1, 1)
            nn.Conv2d(in_channels, reduced_channels, 1),  # Excitation: kompresja liniowa kanałów
            activation,  # Zachowano: Dynamiczna funkcja aktywacji
            nn.Conv2d(reduced_channels, in_channels, 1),  # Excitation: odtworzenie pierwotnego wymiaru kanałów
            nn.Sigmoid()  # Skalowanie wag uwagi do przedziału [0, 1]
        )

    def forward(self, x):
        """
        Definiuje przepływ danych (Forward Pass) - nałożenie wag uwagi na tensor wejściowy.
        """
        # Wykorzystanie natywnego dopasowania wymiarów 4D splotu Conv2d
        return x * self.se(x)


class StochasticDepth(nn.Module):
    """
    Moduł Stochastic Depth (Drop Path).
    Zaawansowana metoda regularyzacji głębokich architektur rezydualnych. Podczas treningu
    losowo dezaktywuje i pomija całe bloki splotowe, przepuszczając sygnał wyłącznie przez
    połączenie skrótu (skip connection), co skutecznie zapobiega przeuczeniu sieci.
    """

    def __init__(self, drop_prob):
        """
        Inicjalizacja modułu gubienia ścieżek rezydualnych.

        Args:
            drop_prob (float): Prawdopodobieństwo wyłączenia całego bloku podczas iteracji.
        """
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        Definiuje przepływ danych z warunkowym pomijaniem warstw.
        """
        # Mechanizm wcześniejszego wyjścia (Early Exit) dla oszczędności pamięci i czasu obliczeń
        if not self.training or self.drop_prob == 0.:
            return x

        keep_prob = 1 - self.drop_prob
        # Dynamiczne wyliczanie kształtu tensora losowego niezależnie od wymiarowości wejścia
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Zaokrąglenie w dół generuje maskę binarną (0 lub 1)

        # Skalowanie tensora wyjściowego w celu zachowania oczekiwanej wartości matematycznej aktywacji
        return (x / keep_prob) * random_tensor

class FusedMBConvBlock(nn.Module):
    """
    Blok Fused-MBConv (EfficientNetV2).
    Zastępuje splot 1x1 i depthwise jednym, standardowym splotem przestrzennym.
    Drastycznie przyspiesza obliczenia na wczesnych etapach sieci na akceleratorach GPU.
    """
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride,
                 use_se, se_reduction, drop_path_rate, activation):
        super(FusedMBConvBlock, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        padding = kernel_size // 2

        # Logika Fused (brak osobnego 1x1 expand):
        if expand_ratio != 1:
            self.fused_conv = ConvBnAct(in_channels, hidden_dim, kernel_size, stride, padding, activation=activation)
            self.project = ConvBnAct(hidden_dim, out_channels, kernel_size=1, activation=None)
        else:
            self.fused_conv = ConvBnAct(in_channels, out_channels, kernel_size, stride, padding, activation=None)
            self.project = nn.Identity()

        self.se = SEBlock(hidden_dim if expand_ratio != 1 else out_channels, se_reduction, activation) if use_se else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fused_conv(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = self.drop_path(x)
            x = x + shortcut
        return x

class ConvBnAct(nn.Module):
    """
    Klasa pomocnicza łącząca operacje splotu, normalizacji oraz aktywacji.
    Hermetyzuje najczęściej występującą sekwencję warstw w jeden reużywalny obiekt,
    co optymalizuje czytelność grafu obliczeniowego sieci i zapobiega błędom implementacyjnym.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False,
                 activation=None):
        """
        Inicjalizacja bloku splotowego Conv-BN-Act.

        Args:
            in_channels (int): Liczba kanałów wejściowych.
            out_channels (int): Liczba kanałów wyjściowych.
            kernel_size (int): Rozmiar jądra splotu.
            stride (int): Krok splotu (stride=2 tnie rozdzielczość przestrzenną o połowę).
            padding (int): Dopełnienie mapy cech.
            groups (int): Grupowanie kanałów (groups=in_channels wymusza splot Depthwise).
            bias (bool): Czy włączyć przesunięcie (domyślnie False, ponieważ BatchNorm je niweluje).
            activation (nn.Module, optional): Obiekt aktywacji. Brak argumentu tworzy warstwę czysto liniową.
        """
        super(ConvBnAct, self).__init__()

        # 1. Definicja bazowych struktur splotowych i stabilizujących
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        ]

        # 2. Opcjonalne dodanie nieliniowości (brak aktywacji wymagany na końcach bloków rezydualnych)
        if activation is not None:
            layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Przepływ danych przez zwartą sekwencję splotową.
        """
        return self.block(x)


class MBConvBlock(nn.Module):
    """
    Blok MBConv (Mobile Inverted Bottleneck Convolution).
    Główna jednostka strukturalna nowoczesnych sieci splotowych (podstawa architektury EfficientNet).
    Przetwarza dane wieloetapowo: poprzez rozszerzenie kanałów (ekspansja 1x1), splot właściwy
    (często Depthwise dla oszczędności FLOPs), mechanizm uwagi (SEBlock) oraz projekcję liniową 1x1.
    """

    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride,
                 use_se, se_reduction, use_depthwise, drop_path_rate, activation):
        """
        Inicjalizacja bloku rezydualnego MBConv.

        Args:
            in_channels (int): Liczba kanałów wejściowych.
            out_channels (int): Liczba kanałów wyjściowych.
            expand_ratio (int): Współczynnik rozszerzenia kanałów wewnątrz bloku.
            kernel_size (int): Wielkość okna splotu właściwego.
            stride (int): Krok splotu (kontroluje downsampling na początku sekcji).
            use_se (bool): Flaga aktywująca moduł uwagi kanałowej.
            se_reduction (int): Współczynnik redukcji bottleneck w SEBlock.
            use_depthwise (bool): Czy zastosować lekki splot separowalny (Depthwise).
            drop_path_rate (float): Współczynnik gubienia bloków dla Stochastic Depth.
            activation (nn.Module): Obiekt dynamicznej aktywacji sieci.
        """
        super(MBConvBlock, self).__init__()

        # Warunek połączenia rezydualnego: zgodność wymiarów wejściowych/wyjściowych oraz krok równy 1
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim

        # Krok 1: Splot rozszerzający (Expansion Phase) - realizowany tanim splotem o oknie 1x1
        if self.expand:
            self.expand_conv = ConvBnAct(in_channels, hidden_dim, kernel_size=1, activation=activation)
        else:
            self.expand_conv = nn.Identity()

        # Krok 2: Splot właściwy z ustawialnym parametrem okna (kernel_size) i krokiem (stride)
        padding = kernel_size // 2

        # Selektywny dobór typu splotu: separowalny (Depthwise) lub pełny splot standardowy
        if use_depthwise:
            self.main_conv = ConvBnAct(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
                                       activation=activation)
        else:
            self.main_conv = ConvBnAct(hidden_dim, hidden_dim, kernel_size, stride, padding, activation=activation)

        # Krok 3: Opcjonalny mechanizm uwagi kanałowej SEBlock
        self.se = SEBlock(hidden_dim, se_reduction, activation) if use_se else nn.Identity()

        # Krok 4: Projekcja liniowa (Pointwise Convolution) - kompresja kanałów do docelowego out_channels
        # BARDZO WAŻNE: Przekazanie activation=None zapewnia matematyczną poprawność przed sumowaniem skip-connection.
        self.project = ConvBnAct(hidden_dim, out_channels, kernel_size=1, activation=None)

        # Krok 5: Opcjonalne wdrożenie regularyzacji DropPath dla operacji rezydualnych
        self.drop_path = StochasticDepth(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        """
        Jawny i ustrukturyzowany przepływ danych przez etapy bloku MBConv.
        """
        shortcut = x

        x = self.expand_conv(x)
        x = self.main_conv(x)
        x = self.se(x)
        x = self.project(x)

        # Dodanie połączenia skrótu (Skip Connection) w przypadku zgodności struktur i wymiarów
        if self.use_residual:
            x = self.drop_path(x)
            x = x + shortcut
        return x


class MyCustomCNN(nn.Module):
    """
    Własna, w pełni parametryzowalna architektura sieci splotowej oparta na blokach MBConv.
    Umożliwia zaawansowane badania nad strukturą modeli poprzez dynamiczne skalowanie głębokości
    (pętle warstw) i szerokości (potęgi dwójki) bezpośrednio z pliku konfiguracyjnego CONFIG.
    """

    def __init__(self, in_channels, num_classes, head_dropout, head_width, head_depth, pooling_mode,
                 custom_base_channels, custom_block_repeats, custom_activation,
                 use_se_block, se_reduction, use_depthwise, drop_path_rate, expansion_ratio, kernel_size):
        """
        Inicjalizacja autorskiej sieci badawczej.

        Args:
            in_channels (int): Liczba kanałów wejściowych obrazu (np. 3 dla RGB).
            num_classes (int): Liczba końcowych klas wyjściowych (kategorii decyzyjnych).
            head_dropout (float): Współczynnik regularyzacji Dropout w klasyfikatorze.
            head_width (int): Liczba neuronów w warstwach ukrytych głowicy.
            head_depth (int): Całkowita liczba warstw liniowych w głowicy.
            pooling_mode (str): Wybór globalnej redukcji przestrzennej ("max" lub "avg").
            custom_base_channels (int): Liczba filtrów dla warstwy wejściowej (Stem).
            custom_block_repeats (list): Lista określająca liczbę bloków w kolejnych sekcjach sieci.
            custom_activation (str): Nazwa globalnej funkcji aktywacji ("silu", "gelu", "relu").
            use_se_block (bool): Globalna flaga włączająca mechanizm uwagi SEBlock.
            se_reduction (int): Współczynnik redukcji bottleneck w SEBlock.
            use_depthwise (bool): Wybór splotów Depthwise Separable zamiast standardowych splotów pełnych.
            drop_path_rate (float): Prawdopodobieństwo wyłączenia warstw w Stochastic Depth.
            expansion_ratio (int): Mnożnik rozszerzania kanałów wewnątrz bloków MBConv.
            kernel_size (int): Rozmiar okna splotu właściwego dla całego ekstraktora.
        """
        super(MyCustomCNN, self).__init__()

        # Rejestracja parametrów wewnętrznych na potrzeby wewnętrznych odwołań
        self.head_dropout = head_dropout
        self.head_width = head_width
        self.head_depth = head_depth
        self.pooling_mode = pooling_mode
        self.custom_base_channels = custom_base_channels
        self.custom_block_repeats = custom_block_repeats
        self.custom_activation = custom_activation
        self.use_se_block = use_se_block
        self.se_reduction = se_reduction
        self.use_depthwise = use_depthwise
        self.drop_path_rate = drop_path_rate
        self.expansion_ratio = expansion_ratio
        self.kernel_size = kernel_size

        # 1. Konfiguracja dynamicznej, globalnej funkcji aktywacji modelu
        if self.custom_activation == "silu":
            self.activation = nn.SiLU()
        elif self.custom_activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # 2. Konfiguracja warstwy Global Pooling (zastępuje destrukcyjne, śródwarstwowe MaxPool2d)
        if self.pooling_mode == "max":
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 3. Inicjalizacja warstwy początkowej (Stem) z krokiem stride=2 dla wstępnego zmniejszenia obrazu
        self.stem = ConvBnAct(in_channels, self.custom_base_channels, kernel_size=3, stride=2, padding=1,
                              activation=self.activation)

        # 4. Dynamiczne budowanie trzonu ekstraktora cech (Feature Extractor) za pomocą pętli
        feature_layers = []
        curr_channels = self.custom_base_channels

        # DODANO: Zmienne do wyliczania liniowego DropPath (Progressive Stochastic Depth)
        total_blocks = sum(self.custom_block_repeats)
        block_counter = 0

        for stage_idx, repeats in enumerate(self.custom_block_repeats):
            out_channels = self.custom_base_channels * (2 ** stage_idx)

            for block_idx in range(repeats):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1

                # Wyliczanie rosnącego prawdopodobieństwa gubienia bloku
                block_counter += 1
                # Używamy block_counter od 1, aby nawet pierwsze bloki miały minimalną (ale niezerową) szansę na drop
                linear_drop_rate = self.drop_path_rate * (block_counter / total_blocks)

                # Wczesne etapy (stage 0) używają szybkiego FusedMBConv, reszta głębokiego MBConv
                if stage_idx == 0:
                    feature_layers.append(
                        FusedMBConvBlock(
                            in_channels=curr_channels,
                            out_channels=out_channels,
                            expand_ratio=self.expansion_ratio,
                            kernel_size=self.kernel_size,
                            stride=stride,
                            use_se=self.use_se_block,
                            se_reduction=self.se_reduction,
                            drop_path_rate=linear_drop_rate,
                            activation=self.activation
                        )
                    )
                else:
                    feature_layers.append(
                        MBConvBlock(
                            in_channels=curr_channels,
                            out_channels=out_channels,
                            expand_ratio=self.expansion_ratio,
                            kernel_size=self.kernel_size,
                            stride=stride,
                            use_se=self.use_se_block,
                            se_reduction=self.se_reduction,
                            use_depthwise=self.use_depthwise,
                            drop_path_rate=linear_drop_rate,
                            activation=self.activation
                        )
                    )
                curr_channels = out_channels

        # Zapakowanie zintegrowanej listy bloków w uniwersalny kontener Sequential
        self.features = nn.Sequential(*feature_layers)

        # 5. Konfiguracja głębokiego klasyfikatora (Głowica MLP) na podstawie wyliczonych dynamicznie kanałów
        in_features = curr_channels
        layers = []
        curr_in = in_features

        # Budowa warstw ukrytych klasyfikatora (wywoływana jeśli head_depth > 1)
        for _ in range(self.head_depth - 1):
            layers.append(nn.Linear(curr_in, self.head_width))
            layers.append(self.activation)
            layers.append(nn.Dropout(self.head_dropout))
            curr_in = self.head_width

        # Finalna warstwa decyzyjna (liniowa) dopasowana do oczekiwanej liczby klas wyjściowych
        layers.append(nn.Dropout(self.head_dropout))
        layers.append(nn.Linear(curr_in, num_classes))

        self.classifier = nn.Sequential(*layers)

        # 6. Przygotowanie pod system wyjaśnialności sztucznej inteligencji (Grad-CAM)
        # Niezależna, dynamiczna referencja wskazująca zawsze na ostatni fizyczny blok ekstraktora cech
        self.last_conv_layer = self.features[-1]

    def forward(self, x):
        """
        Definiuje przepływ danych przez kompletną, zrefaktoryzowaną sieć splotową (Forward Pass).
        """
        x = self.stem(x)
        x = self.features(x)
        x = self.global_pool(x)
        x = x.reshape(x.shape[0], -1)  # Spłaszczenie tensora przestrzennego 1x1 do płaskiego wektora cech
        x = self.classifier(x)
        return x