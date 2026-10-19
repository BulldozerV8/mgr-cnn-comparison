import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
import re
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# =================================================================
# KONFIGURACJA STYLU AKADEMICKIEGO
# =================================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
OUTPUT_BASE = "Thesis_Outputs"

WHITELIST_MATRICES = [
    'FINAL_Ptaki_448px_CV5',
    'BASELINE_Ptaki_Water',
    'SCALABILITY',
    'ROBUSTNESS_Weather',
    'FINAL_Samoloty_448px_CV5'
]

RADAR_MAP_PTAKI = {
    'FINAL_Ptaki_448px_CV5': 'Baseline (448px)',
    'BASELINE_Ptaki_Water_448px_CV5': 'Drugi podzbiór (Wodne)',
    'SCALABILITY_CUB_Full_200klas': 'Skalowalność (200 klas)',
    'ROBUSTNESS_Weather_FINAL_Ptaki_Fold1': 'Odporność (Pogoda)'
}

RADAR_MAP_SAMOLOTY = {
    'FINAL_Samoloty_448px_CV5': 'Baseline (448px)',
    'ROBUSTNESS_Weather_FINAL_Samoloty_Fold1': 'Odporność (Pogoda)'
}

# Słowniki do tłumaczenia nazw w tuningu na język polski do pracy
PARAM_LABELS = {
    'batch_size': 'Rozmiar paczki (Batch Size)',
    'BatchSize': 'Rozmiar paczki (Batch Size)',
    'learning_rate': 'Współczynnik uczenia (Learning Rate)',
    'LR': 'Współczynnik uczenia (Learning Rate)',
    'unfreeze_blocks': 'Liczba odmrożonych bloków',
    'Unfreeze': 'Liczba odmrożonych bloków',
    'pooling_mode': 'Metoda agregacji (Pooling)',
    'Pooling': 'Metoda agregacji (Pooling)',
    'head_dropout': 'Współczynnik Dropout',
    'head_width_and_depth': 'Szerokość i głębokość głowy',
    'dropout_and_weight_decay': 'Dropout i Weight Decay',
    'Reg_Dr_WD': 'Regularyzacja (Dropout / Weight Decay)',
    'Head_W_D': 'Struktura głowy (Szerokość / Głębokość)',
    'num_samples_per_epoch': 'Liczba próbek na epokę',
    'Samples': 'Liczba próbek na epokę'
}

VALUE_LABELS = {
    'avg': 'Średnia (Avg)',
    'max': 'Maksymalna (Max)',
    'gem': 'Geometryczna (GeM)',
    'pooling_mode_avg': 'Średnia (Avg)',
    'pooling_mode_max': 'Maksymalna (Max)',
    'pooling_mode_gem': 'Geometryczna (GeM)'
}


# =================================================================
# FUNKCJA DO RADARÓW (OFICJALNA Z MATPLOTLIB)
# =================================================================
def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")

        def _gen_axes_spines(self):
            if frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle', path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            return super()._gen_axes_spines()

    register_projection(RadarAxes)
    return theta


# =================================================================
# GŁÓWNA KLASA ANALITYCZNA
# =================================================================
class ThesisAnalyzer:
    def __init__(self, manifest_path):
        self.df = pd.read_csv(manifest_path)
        self.create_dirs()

    def create_dirs(self):
        paths = ["5_1_Skutecznosc", "5_3_Efektywnosc", "5_4_Tuning", "5_6_Analiza_Bledow", "Tabele_Szczegolowe"]
        for p in paths:
            os.makedirs(os.path.join(OUTPUT_BASE, p), exist_ok=True)

    def get_raw_data(self, folder_group, subfolder, file_source):
        path = os.path.join("../results", folder_group, subfolder, file_source)
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    # --- SEKCJA 5.1: WYKRES SŁUPKOWY ZGRUPOWANY (NAPRAWIONY Z ODCZYTEM SUMMARY) ---
    def process_comparison_bars(self):
        print("Generowanie słupków porównawczych metryk (5.1)...")
        metrics = ['test_acc', 'test_prec', 'test_rec', 'test_f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        for domain_name, radar_map in [("Ptaki", RADAR_MAP_PTAKI), ("Samoloty", RADAR_MAP_SAMOLOTY)]:
            all_fold_results = []

            for test_id, display_label in radar_map.items():
                matches = self.df[self.df['test_name'].str.contains(test_id)]
                if matches.empty: continue
                row = matches.iloc[0]

                raw = self.get_raw_data(row['folder_group'], row['subfolder'], row['file_source'])
                if raw is not None:
                    folds = raw[raw['type'] == 'SUMMARY'][['fold'] + metrics].copy()

                    for m in metrics:
                        folds[m] = pd.to_numeric(folds[m], errors='coerce')
                        folds[m] = folds[m].apply(lambda x: x * 100 if pd.notna(x) and x <= 1.0 else x)

                    melted = folds.melt(id_vars='fold', value_vars=metrics, var_name='Metric', value_name='Score')
                    melted['Zbiór danych'] = display_label
                    all_fold_results.append(melted)

            if all_fold_results:
                plot_df = pd.concat(all_fold_results)
                plot_df['Metric'] = plot_df['Metric'].replace(dict(zip(metrics, metric_labels)))

                plt.figure(figsize=(12, 7))
                ax = sns.barplot(data=plot_df, x='Metric', y='Score', hue='Zbiór danych',
                                 palette="muted", errorbar='sd', capsize=.1)

                plt.ylabel("Wynik (%)")
                plt.xlabel("Metryka")
                plt.ylim(max(0, plot_df['Score'].min() - 5), 100)
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(OUTPUT_BASE, "5_1_Skutecznosc", f"Porownanie_Metryk_{domain_name}.png"))
                plt.close()

    # --- SEKCJA 5.4: BOX PLOTY ---
    def process_boxplots(self):
        print("Generowanie Box Plotów dla foldów (5.4)...")
        cv_tests = self.df[(self.df['use_cv'] == True) & (~self.df['folder_group'].str.contains('tuning', case=False))]

        for folder_grp in cv_tests['folder_group'].unique():
            domain_name = "Ptaki" if "ptaki" in folder_grp.lower() else "Samoloty"
            tests_in_domain = cv_tests[cv_tests['folder_group'] == folder_grp]

            box_data = []
            for _, row in tests_in_domain.iterrows():
                raw_data = self.get_raw_data(row['folder_group'], row['subfolder'], row['file_source'])
                if raw_data is not None:
                    folds = raw_data[raw_data['type'] == 'EPOCH'].groupby('fold')['val_acc'].max().reset_index()
                    folds['test_name'] = row['test_name'].replace('FINAL_', '').replace('_CV5', '')
                    folds['val_acc'] = folds['val_acc'] * 100 if folds['val_acc'].max() <= 1.0 else folds['val_acc']
                    box_data.append(folds)

            if box_data:
                combined_box = pd.concat(box_data)
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='test_name', y='val_acc', hue='test_name', data=combined_box, palette="Set2",
                            legend=False)
                sns.swarmplot(x='test_name', y='val_acc', data=combined_box, color=".25")
                plt.title(f"Stabilność wyników (Rozkład Accuracy z 5 foldów) - {domain_name}")
                plt.ylabel("Accuracy Walidacyjne (%)")
                plt.xlabel("Eksperyment")
                plt.xticks(rotation=15)

                plt.savefig(os.path.join(OUTPUT_BASE, "5_4_Tuning", f"BoxPlot_Stabilnosc_{domain_name}.png"))
                plt.close()

    # --- SEKCJA 5.3: EFEKTYWNOŚĆ ---
    def process_efficiency(self):
        print("Analiza efektywności: Accuracy vs Czas z odchyleniami (5.3)...")
        eff_df = self.df[self.df['test_name'].str.contains('224|448|BASELINE', case=False)].copy()

        if not eff_df.empty:
            scatter_data = []
            for _, row in eff_df.iterrows():
                raw_data = self.get_raw_data(row['folder_group'], row['subfolder'], row['file_source'])
                if raw_data is not None:
                    fold_accs = raw_data[raw_data['type'] == 'EPOCH'].groupby('fold')['val_acc'].max() * 100
                    fold_times = raw_data[raw_data['type'] == 'EPOCH'].groupby('fold')['train_time_sec'].sum()

                    scatter_data.append({
                        'Model': row['dataset_tag'],
                        'Rozdzielczość': '448px' if '448' in row['test_name'] else '224px',
                        'Mean_Acc': fold_accs.mean(),
                        'Std_Acc': fold_accs.std(),
                        'Mean_Time': fold_times.mean(),
                        'Std_Time': fold_times.std()
                    })

            plot_df = pd.DataFrame(scatter_data)

            plt.figure(figsize=(10, 6))
            for i, row in plot_df.iterrows():
                plt.errorbar(x=row['Mean_Time'], y=row['Mean_Acc'],
                             xerr=row['Std_Time'], yerr=row['Std_Acc'],
                             fmt='o', markersize=10, capsize=5, label=f"{row['Model']} ({row['Rozdzielczość']})")

            plt.xlabel("Średni czas uczenia 1 folda (sekundy)")
            plt.ylabel("Średnia Dokładność (%)")
            plt.title("Kompromis: Czas Uczenia vs Skuteczność (z odchyleniami std)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(OUTPUT_BASE, "5_3_Efektywnosc", "Accuracy_vs_TrainingTime_ErrorBars.png"))
            plt.close()

    # --- SEKCJA 5.4: KRZYWE UCZENIA (LOSS + ACCURACY OBOK SIEBIE) ---
    def process_learning_curves(self):
        print("Generowanie podwójnych krzywych uczenia (Loss + Accuracy) (5.4)...")
        cv_tests = \
        self.df[(self.df['use_cv'] == True) & (~self.df['folder_group'].str.contains('tuning', case=False))][
            'subfolder'].unique()

        for test in cv_tests:
            subset = self.df[self.df['subfolder'] == test]
            all_epochs = []
            for _, row in subset.iterrows():
                e_data = self.get_raw_data(row['folder_group'], row['subfolder'], row['file_source'])
                if e_data is not None:
                    e_data = e_data[e_data['type'] == 'EPOCH'].copy()

                    # Konwersja i skalowanie Accuracy
                    if e_data['train_acc'].max() <= 1.0: e_data['train_acc'] *= 100
                    if e_data['val_acc'].max() <= 1.0: e_data['val_acc'] *= 100

                    # Konwersja Loss na typ numeryczny
                    e_data['train_loss'] = pd.to_numeric(e_data['train_loss'], errors='coerce')
                    e_data['val_loss'] = pd.to_numeric(e_data['val_loss'], errors='coerce')
                    e_data['epoch'] = pd.to_numeric(e_data['epoch'], errors='coerce')

                    all_epochs.append(e_data)
            if not all_epochs: continue

            combined = pd.concat(all_epochs).dropna(subset=['epoch']).sort_values('epoch')

            # Agregacja średniej i odchylenia dla wszystkich 4 metryk
            stats = combined.groupby('epoch').agg({
                'train_loss': ['mean', 'std'], 'val_loss': ['mean', 'std'],
                'train_acc': ['mean', 'std'], 'val_acc': ['mean', 'std']
            }).reset_index()

            # Tworzenie siatki 1x2 (jeden rząd, dwie kolumny)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 1. WYKRES LEWY: Funkcja Kosztu (Loss)
            ax1.plot(stats['epoch'], stats['train_loss']['mean'], label='Trening (Loss)', color=COLORS[3],
                     marker='o')
            ax1.fill_between(stats['epoch'], stats['train_loss']['mean'] - stats['train_loss']['std'],
                             stats['train_loss']['mean'] + stats['train_loss']['std'], color=COLORS[3], alpha=0.15)
            ax1.plot(stats['epoch'], stats['val_loss']['mean'], label='Walidacja (Loss)', color=COLORS[4],
                     marker='o')
            ax1.fill_between(stats['epoch'], stats['val_loss']['mean'] - stats['val_loss']['std'],
                             stats['val_loss']['mean'] + stats['val_loss']['std'], color=COLORS[4], alpha=0.15)

            ax1.set_xlabel("Epoka")
            ax1.set_ylabel("Wartość Loss")
            ax1.set_title("Funkcja kosztu (Loss Curve)")
            ax1.set_xticks(range(int(stats['epoch'].min()), int(stats['epoch'].max()) + 1))
            ax1.set_ylim(bottom=0)  # Loss zaczyna się od zera w górę
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle='--', alpha=0.5)

            # 2. WYKRES PRAWY: Dokładność (Accuracy)
            ax2.plot(stats['epoch'], stats['train_acc']['mean'], label='Trening (Acc)', color=COLORS[0], marker='o')
            ax2.fill_between(stats['epoch'], stats['train_acc']['mean'] - stats['train_acc']['std'],
                             stats['train_acc']['mean'] + stats['train_acc']['std'], color=COLORS[0], alpha=0.15)
            ax2.plot(stats['epoch'], stats['val_acc']['mean'], label='Walidacja (Acc)', color=COLORS[1], marker='o')
            ax2.fill_between(stats['epoch'], stats['val_acc']['mean'] - stats['val_acc']['std'],
                             stats['val_acc']['mean'] + stats['val_acc']['std'], color=COLORS[1], alpha=0.15)

            ax2.set_xlabel("Epoka")
            ax2.set_ylabel("Dokładność (%)")
            ax2.set_title("Dokładność (Accuracy Curve)")
            ax2.set_xticks(range(int(stats['epoch'].min()), int(stats['epoch'].max()) + 1))
            ax2.set_ylim(0, 100)  # Twarda, naukowa skala 0-100%
            ax2.legend(loc='lower right')
            ax2.grid(True, linestyle='--', alpha=0.5)

            # Dopasowanie układu i zapis
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_BASE, "5_4_Tuning", f"Krzywa_Uczenia_{test}.png"))
            plt.close()

    # --- SEKCJA 5.4: TUNING BARS (PRZETŁUMACZONE PODPISY) ---
    def process_tuning_bars(self):
        print("Generowanie wykresów tuningu (5.4)...")
        tuning_groups = self.df[self.df['folder_group'].str.contains('tuning', case=False)].copy()

        def extract_param(name):
            base = re.sub(r'^Tune_', '', name)
            base = re.sub(r'_Trial_\d+$', '', base)
            if 'head_width' in base and 'head_depth' in base: return 'head_width_and_depth', base
            if 'head_dropout' in base and 'weight_decay' in base: return 'dropout_and_weight_decay', base
            if 'Reg_Dr' in base: return 'Reg_Dr_WD', base
            if 'Head_W' in base: return 'Head_W_D', base
            parts = base.rsplit('_', 1)
            if len(parts) == 2: return parts[0], parts[1]
            return "Inne", base

        tuning_groups[['param_name', 'param_value']] = tuning_groups['subfolder'].apply(
            lambda x: pd.Series(extract_param(x)))

        for folder_grp in tuning_groups['folder_group'].unique():
            domain_name = "Ptaki" if "ptaki" in folder_grp.lower() else "Samoloty"
            domain_data = tuning_groups[tuning_groups['folder_group'] == folder_grp]

            for param in domain_data['param_name'].unique():
                if param == "Inne": continue
                subset = domain_data[domain_data['param_name'] == param].sort_values(by='param_value').copy()
                plt.figure(figsize=(10, 6))
                plot_data = subset['test_acc'] * 100 if subset['test_acc'].max() <= 1.0 else subset['test_acc']

                # POPRAWKA: Tłumaczenie wartości na osi X (np. avg -> Średnia (Avg))
                subset['display_value'] = subset['param_value'].map(lambda x: VALUE_LABELS.get(x, x))

                sns.barplot(x=subset['display_value'], y=plot_data, hue=subset['display_value'], palette="Blues_d",
                            legend=False)
                plt.xticks(rotation=45, ha='right')

                # POPRAWKA: Przetłumaczona czysta etykieta osi X ze słownika mapowania
                clean_xlabel = PARAM_LABELS.get(param, f"Wartość dla {param}")
                plt.xlabel(clean_xlabel)

                plt.ylabel("Accuracy (%)")
                plt.ylim(plot_data.min() * 0.95, min(plot_data.max() * 1.05, 100.0))
                plt.savefig(os.path.join(OUTPUT_BASE, "5_4_Tuning", f"Tuning_{domain_name}_{param}.png"))
                plt.close()

    # --- SEKCJA 5.6: MACIERZE ---
    def collect_confusion_matrices(self):
        print("Kopiowanie macierzy pomyłek (5.6)...")
        for _, row in self.df.iterrows():
            if not any(wl in row['test_name'] for wl in WHITELIST_MATRICES): continue
            src_dir = os.path.join("../results", str(row['folder_group']), str(row['subfolder']), "plots")
            if os.path.exists(src_dir):
                for f in os.listdir(src_dir):
                    if "confusion_matrix" in f:
                        shutil.copy2(os.path.join(src_dir, f), os.path.join(OUTPUT_BASE, "5_6_Analiza_Bledow",
                                                                            f"{row['dataset_tag']}_{row['subfolder']}_{f}"))

    # --- EKSPORT OSOBNYCH TABEL (ZAOKRĄGLONE DO 2 MIEJSC PO PRZECINKU) ---
    def export_individual_tables(self):
        print("Eksportowanie osobnych tabel szczegółowych do Worda...")
        for _, row in self.df.iterrows():
            if not any(wl in row['test_name'] for wl in WHITELIST_MATRICES):
                continue

            raw_data = self.get_raw_data(row['folder_group'], row['subfolder'], row['file_source'])
            if raw_data is not None:
                table_data = raw_data[raw_data['type'].isin(['FOLD_SUMMARY', 'SUMMARY'])].copy()

                if 'FOLD_SUMMARY' not in table_data['type'].values:
                    last_epochs = raw_data[raw_data['type'] == 'EPOCH'].groupby('fold').tail(1)
                    table_data = pd.concat([last_epochs, table_data])

                cols_to_keep = ['type', 'fold', 'test_acc', 'test_prec', 'test_rec', 'test_f1', 'train_time_sec',
                                'inference_ms']
                existing_cols = [c for c in cols_to_keep if c in table_data.columns]
                clean_table = table_data[existing_cols].copy()

                # POPRAWKA: Przemnożenie, rzutowanie na liczby oraz sztywne zaokrąglenie do 2 miejsc po przecinku (.round(2))
                metrics_to_scale = ['test_acc', 'test_prec', 'test_rec', 'test_f1']
                for m in metrics_to_scale:
                    if m in clean_table.columns:
                        clean_table[m] = pd.to_numeric(clean_table[m], errors='coerce')
                        clean_table[m] = (
                            clean_table[m].apply(lambda x: x * 100 if pd.notna(x) and x <= 1.0 else x)).round(2)

                filename = f"Tabela_{row['test_name']}_Szczegoly.csv"
                clean_table.to_csv(os.path.join(OUTPUT_BASE, "Tabele_Szczegolowe", filename), index=False, sep=';')


# =================================================================
# URUCHOMIENIE
# =================================================================
if __name__ == "__main__":
    if os.path.exists("master_manifest.csv"):
        analyzer = ThesisAnalyzer("master_manifest.csv")
        analyzer.process_comparison_bars()
        analyzer.process_boxplots()
        analyzer.process_learning_curves()
        analyzer.process_tuning_bars()
        analyzer.process_efficiency()
        analyzer.collect_confusion_matrices()
        analyzer.export_individual_tables()
        print("\nGOTOWE! Prawdziwe, naukowe wykresy czekają w folderze 'Thesis_Outputs'.")
    else:
        print("BŁĄD: Nie znaleziono pliku master_manifest.csv!")