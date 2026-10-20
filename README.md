
# Klasyfikacja obiektów FGVC: EfficientNet-B0 vs autorska sieć CNN

Zautomatyzowane środowisko badawcze do drobnoziarnistej klasyfikacji obrazów (Fine-Grained Visual Categorization, FGVC), opracowane w ramach pracy magisterskiej. Celem projektu jest porównanie modelu EfficientNet-B0 wykorzystującego uczenie transferowe (transfer learning) z autorską siecią konwolucyjną MyCustomCNN, trenowaną od podstaw.

## 🚀 Główne funkcjonalności

- **Porównanie dwóch architektur:** EfficientNet-B0 z wagami wstępnie wytrenowanymi na zbiorze ImageNet oraz autorska sieć MyCustomCNN, wykorzystująca bloki Fused-MBConv i MBConv, mechanizm uwagi kanałowej SEBlock oraz regularyzację DropPath.
- **Przetwarzanie i augmentacja obrazów:**
  - Dopasowanie proporcji obrazu do wymaganego rozmiaru z wykorzystaniem własnej transformacji `SquarePad`.
  - Augmentacja danych treningowych, m.in. za pomocą `TrivialAugmentWide` i `RandomErasing`.
  - Symulacja wybranych zakłóceń pogodowych — deszczu, mgły i śniegu — za pomocą `WeatherTransformWrapper`, wykorzystującego bibliotekę Albumentations.
- **Obsługa niezbalansowanych klas:** Ważone próbkowanie z wykorzystaniem `WeightedRandomSampler`, z kontrolowaną liczbą próbek w epoce.
- **Trening i walidacja:**
  - Automatyczne wykorzystanie automatycznej mieszanej precyzji (Automatic Mixed Precision, AMP) z `GradScaler` w celu ograniczenia zużycia pamięci GPU.
  - Walidacja krzyżowa z użyciem `StratifiedKFold` lub walidacja typu hold-out.
  - Wczesne zatrzymywanie treningu (`EarlyStopping`) na podstawie straty walidacyjnej (validation loss).
- **Strojenie hiperparametrów:** Moduł `optimization/autotune.py` wykorzystuje bibliotekę Optuna i sampler `GridSampler` do przeszukiwania zdefiniowanej siatki konfiguracji.
- **Ewaluacja i wyjaśnialność:**
  - Generowanie map Grad-CAM przedstawiających obszary obrazu wpływające na predykcję modelu.
  - Generowanie macierzy pomyłek, w tym macierzy znormalizowanych.
  - Pomiar czasu treningu i szczytowego zużycia pamięci GPU.
  - Analiza wyników oraz generowanie wykresów i zestawień.
- **Interfejs graficzny:** Aplikacja oparta na `tkinter`, umożliwiająca interaktywną klasyfikację nowych obrazów za pomocą wytrenowanego modelu.

## 🛠 Technologie i środowisko

### Oprogramowanie

- **Język:** Python 3.13.5.
- **IDE:** PyCharm 2025.2.3.
- **Framework uczenia głębokiego:** PyTorch.
- **Obsługa GPU:** CUDA i NVIDIA cuDNN.
- **Optymalizacja obliczeń:** Włączona opcja `torch.backends.cudnn.benchmark`, dobierająca algorytmy operacji konwolucyjnych do rozmiarów wejściowych.

### Główne biblioteki

- `torchvision` — architektury modeli i transformacje obrazów.
- `scikit-learn` — podział danych i walidacja krzyżowa.
- `albumentations` — augmentacja obrazów i symulacja zakłóceń pogodowych.
- `torchmetrics` — obliczanie metryk klasyfikacyjnych.
- `optuna` — strojenie hiperparametrów.
- `opencv-python` — operacje na obrazach i przygotowywanie wizualizacji Grad-CAM.
- `tkinter` i `PIL.ImageTk` — interfejs graficzny.
- `pandas` i `numpy` — przetwarzanie danych i obliczenia numeryczne.
- `psutil` — monitorowanie zasobów systemowych.
- `thop` — szacowanie liczby operacji wykonywanych przez model.

### Sprzęt wykorzystany w badaniach

- **GPU:** NVIDIA GeForce RTX 5070 Ti, 16 GB VRAM.
- **CPU:** AMD Ryzen 7 7800X3D.
- **RAM:** 32 GB.

Podane wersje oprogramowania i parametry sprzętowe opisują środowisko wykorzystane w badaniach. Nie stanowią gwarancji identycznych wyników na innych konfiguracjach.

## 📂 Struktura projektu

Projekt podzielono na moduły odpowiedzialne za obsługę danych, modele, trening, ewaluację i narzędzia pomocnicze.

| Ścieżka | Opis |
|---|---|
| `data/` | `data_manager.py` — obsługa danych i ważone próbkowanie; `transforms.py` — transformacje i augmentacja obrazów. |
| `models/` | `cnn_base.py` — obsługa architektur bazowych; `custom_model.py` — definicja autorskiej sieci MyCustomCNN. |
| `training/` | `training_engine.py` — pętla treningowa i walidacyjna; `callbacks.py` — mechanizmy wczesnego zatrzymywania. |
| `evaluation/` | `evaluator.py` — obliczanie metryk; `visualization.py` — wykresy i macierze pomyłek; `gradcam.py` — generowanie map Grad-CAM. |
| `utils/` | `logger.py` — logowanie wyników; `helpers.py` — funkcje pomocnicze, w tym obsługa powtarzalności eksperymentów. |
| `optimization/` | `autotune.py` i `tuner.py` — skrypty do strojenia hiperparametrów. |
| `prediction/` | `predictor.py` — obsługa predykcji; `run_cli.py` — uruchamianie z wiersza poleceń; `run_gui.py` — interfejs graficzny. |
| `config.py` | Konfiguracja architektury, hiperparametrów, przetwarzania danych i przebiegu eksperymentów. |
| `experiment.py` | Obsługa pojedynczego eksperymentu, w tym treningu, ewaluacji i zapisu wyników. |
| `main.py` | Główny punkt wejścia do uruchamiania eksperymentów. |
| `run_final_evaluations.py` | Skrypt do uruchamiania zdefiniowanych końcowych ewaluacji. |
| `old_scripts/` | Starsze skrypty i wcześniejsze wersje narzędzi eksperymentalnych. |

Dodatkowe skrypty w katalogu głównym służą m.in. do analizy wyników, obliczania FLOPs, generowania wykresów, analizy statystycznej i podsumowywania czasu eksperymentów.

## 📊 Wykorzystane zbiory danych

1. **CUB-200-2011** — zbiór obrazów ptaków wykorzystywany w zadaniach drobnoziarnistej klasyfikacji.
2. **Military Aircraft Detection Dataset** — zbiór obrazów samolotów wojskowych, poddany autorskiemu czyszczeniu obejmującemu usuwanie obrazów zaszumionych, o bardzo niskiej rozdzielczości lub nieprawidłowo skadrowanych.

Eksperymenty obejmowały różne konfiguracje danych, w tym:

- podzbiór 20 klas ptaków w rozdzielczościach 448 × 448 i 224 × 224;
- podzbiór ptaków wodnych;
- podzbiór 20 klas samolotów;
- testy z zakłóceniami pogodowymi;
- większe zadania klasyfikacyjne obejmujące 200 klas ptaków i 101 klas samolotów.

**Dane wejściowe:** Zbiory danych należy przygotować we własnym zakresie i umieścić w lokalizacjach zgodnych z konfiguracją projektu. Nie należy zakładać, że obrazy są częścią repozytorium.

## 🏆 Wyniki eksperymentów

W przeprowadzonych eksperymentach EfficientNet-B0 osiągał wyższą dokładność klasyfikacji i wymagał krótszego czasu treningu niż autorska sieć MyCustomCNN.

Według wyników analizy statystycznej przedstawionej w pracy magisterskiej porównanie obejmowało osiem podzbiorów danych. W siedmiu przypadkach różnice między modelami uznano za istotne statystycznie przy poziomie istotności α = 0,05. Dla podzbioru 20 klas ptaków w rozdzielczości 224 × 224 nie stwierdzono istotnej statystycznie różnicy (p = 0,1402).

### Porównanie dokładności, czasu treningu i liczby operacji

| Model | Zbiór danych | Średnia dokładność (Accuracy) | Średni czas treningu na fold | FLOPs (G) |
|---|---|---:|---:|---:|
| EfficientNet-B0 | Ptaki — 20 klas, 448 × 448 | 92,62% | ~8 min | 1,898 |
| MyCustomCNN | Ptaki — 20 klas, 448 × 448 | 82,50% | ~66 min | 23,115 |
| EfficientNet-B0 | Samoloty — 20 klas, 448 × 448 | 91,08% | ~37 min | 1,898 |
| MyCustomCNN | Samoloty — 20 klas, 448 × 448 | 65,87% | ~111 min | 23,116 |

### Wyniki dla większej liczby klas

- **EfficientNet-B0:** uzyskał dokładność 81,06% w zadaniu klasyfikacji 200 klas ptaków oraz 85,62% w zadaniu klasyfikacji 101 klas samolotów. W przeprowadzonych eksperymentach wykazywał również lepsze wyniki w testach z zakłóceniami pogodowymi. Mapy Grad-CAM wskazywały na bardziej spójne obszary aktywacji związane z klasyfikowanym obiektem.
- **MyCustomCNN:** sieć posiadała około 1,15 mln parametrów, lecz w przedstawionych eksperymentach osiągała niższą dokładność i wymagała dłuższego treningu. Wyniki wskazywały na większą podatność na wpływ tła oraz spadek skuteczności w trudniejszych warunkach i przy większej liczbie klas. Mapy Grad-CAM wskazywały na bardziej rozproszone obszary aktywacji.

### Interpretacja wyników

Wyniki dotyczą konkretnych zbiorów danych, podziałów i konfiguracji eksperymentalnych. Nie należy traktować ich jako gwarancji uzyskania takiej samej skuteczności w innych zadaniach.

Istotność statystyczna nie jest równoznaczna z dużą wartością praktyczną różnicy ani nie dowodzi uniwersalnej przewagi modelu. Wnioski należy interpretować z uwzględnieniem zastosowanej procedury porównawczej, liczby obserwacji i założeń wykorzystanych testów statystycznych.

## ⚙️ Instalacja i uruchomienie

### 1. Pobranie repozytorium

```bash
git clone <adres_repozytorium>
cd <nazwa_folderu_projektu>
```

Zastąp wartości w nawiasach ostrych adresem repozytorium i nazwą utworzonego katalogu.

### 2. Przygotowanie środowiska

Zaleca się utworzenie odizolowanego środowiska Python, np. za pomocą `venv`:

```bash
python -m venv .venv
```

Aktywacja środowiska w systemie Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Aktywacja środowiska w systemie Linux lub macOS:

```bash
source .venv/bin/activate
```

Następnie należy zainstalować zależności projektu w wersjach zgodnych z używaną wersją Pythona, PyTorch i środowiskiem CUDA. Wersję PyTorch obsługującą odpowiednią konfigurację GPU należy dobrać zgodnie z oficjalnymi instrukcjami instalacji PyTorch.

### 3. Konfiguracja

Przed uruchomieniem projektu należy:

1. Pobrać i przygotować wymagane zbiory danych.
2. Ustawić ścieżki do danych w `config.py`.
3. Wybrać architekturę modelu za pomocą parametru `model_name`.
4. Skonfigurować rozdzielczość obrazów, parametry treningu oraz pozostałe opcje eksperymentu.
5. Ustawić tryb ewaluacji za pomocą `use_cv` — walidację krzyżową albo tryb bez walidacji krzyżowej.

Dostępne parametry konfiguracyjne obejmują również m.in. augmentację, balansowanie klas, hiperparametry treningu oraz ustawienia autorskiej architektury.

### 4. Uruchomienie treningu i ewaluacji

Z katalogu głównego projektu:

```bash
python main.py
```

### 5. Uruchomienie interfejsu graficznego

Po przygotowaniu wytrenowanego modelu i wymaganych metadanych można uruchomić interfejs do klasyfikacji nowych obrazów:

```bash
python prediction/run_gui.py
```

## 📈 Analiza wyników

Projekt zawiera dodatkowe skrypty wspomagające analizę eksperymentów, w tym:

- zestawianie wyników w arkuszach kalkulacyjnych;
- generowanie krzywych uczenia;
- analizę rozkładu klas przed balansowaniem i po nim;
- szacowanie liczby operacji FLOPs i liczby parametrów modeli;
- analizę statystyczną wyników;
- zestawianie czasu treningu.

Wyniki i artefakty eksperymentów są zapisywane w katalogach wynikowych zgodnie z konfiguracją i przebiegiem eksperymentu.
