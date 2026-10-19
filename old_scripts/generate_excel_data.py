import os
import pandas as pd
import numpy as np
import re

OUTPUT_FILE = "../Thesis_Data_For_Excel.xlsx"

# Sztywna, profesjonalna kolejność sekcji z Twojego planu pracy magisterskiej
BIRDS_ORDER = [
    'Baseline Ptaki (448px)',
    'Ptaki Wodne',
    'Skalowalność (200 klas)',
    'Robustness Weather (Pogoda)',
    'Robustness Weather (Drugi podzbiór – Ptaki Wodne)',
    'Model bazowy (224px)'
]

PLANES_ORDER = [
    'Baseline Samoloty (448px)',
    'Samoloty (Drugi podzbiór)',
    'Skalowalność (Pełna liczba klas samolotów)',
    'Robustness Weather (Pogoda)',
    'Robustness Weather (Drugi podzbiór – Samoloty)',
    'Model bazowy (224px)'
]

VALUE_LABELS = {
    'avg': 'Średnia (Avg)', 'max': 'Maksymalna (Max)', 'gem': 'Geometryczna (GeM)',
    'pooling_mode_avg': 'Średnia (Avg)', 'pooling_mode_max': 'Maksymalna (Max)',
    'pooling_mode_gem': 'Geometryczna (GeM)'
}


class ExcelDataExtractor:
    def __init__(self, manifest_path):
        self.df = pd.read_csv(manifest_path)
        self.writer = pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl')

    # Pancerny, automatyczny unifikator nazw eksperymentów dla WSZYSTKICH sekcji
    def classify_row(self, row):
        t = (str(row.get('test_name', '')) + " " + str(row.get('subfolder', ''))).lower()

        if any(w in t for w in ['ptaki', 'cub', 'bird']):
            domain = "Ptaki"
            if 'water' in t and ('weather' in t or 'pogoda' in t):
                return domain, 'Robustness Weather (Drugi podzbiór – Ptaki Wodne)'
            if 'water' in t or 'wodne' in t:
                return domain, 'Ptaki Wodne'
            if 'scale' in t or '200' in t:
                return domain, 'Skalowalność (200 klas)'
            if 'weather' in t or 'pogoda' in t:
                return domain, 'Robustness Weather (Pogoda)'
            if '448' in t or 'baseline' in t:
                return domain, 'Baseline Ptaki (448px)'
            if '224' in t:
                return domain, 'Model bazowy (224px)'
            return domain, row['subfolder']

        elif any(w in t for w in ['samoloty', 'aircraft', 'fgvc', 'plane']):
            domain = "Samoloty"
            if 'weather' in t and ('sub' in t or 'drugi' in t or 'subset' in t):
                return domain, 'Robustness Weather (Drugi podzbiór – Samoloty)'
            if 'weather' in t or 'pogoda' in t:
                return domain, 'Robustness Weather (Pogoda)'
            if 'scale' in t or 'full' in t or '100' in t:
                return domain, 'Skalowalność (Pełna liczba klas samolotów)'
            if 'sub' in t or 'drugi' in t or 'subset' in t:
                return domain, 'Samoloty (Drugi podzbiór)'
            if '448' in t or 'baseline' in t:
                return domain, 'Baseline Samoloty (448px)'
            if '224' in t:
                return domain, 'Model bazowy (224px)'
            return domain, row['subfolder']

        return "Inne", row['subfolder']

    def get_raw_data(self, row):
        path = os.path.join("../results", row['folder_group'], row['subfolder'], row['file_source'])
        return pd.read_csv(path) if os.path.exists(path) else None

    # --- 5.1: METRYKI TESTOWE (ZACHOWANA KOLEJNOŚĆ PLANU) ---
    def extract_section_5_1(self):
        print("Przygotowanie danych dla sekcji 5.1 (Skuteczność)...")
        metrics = ['test_acc', 'test_prec', 'test_rec', 'test_f1']
        main_tests = self.df[~self.df['folder_group'].str.contains('tuning', case=False)].copy()

        birds_rows = []
        planes_rows = []

        for _, row in main_tests.drop_duplicates('subfolder').iterrows():
            domain, clean_name = self.classify_row(row)
            raw = self.get_raw_data(row)

            if raw is not None:
                folds = raw[raw['type'] == 'SUMMARY'].copy()
                for m in metrics:
                    folds[m] = pd.to_numeric(folds[m], errors='coerce')
                    folds[m] = folds[m].apply(lambda x: x * 100 if pd.notna(x) and x <= 1.0 else x)

                res_dict = {
                    'Eksperyment': clean_name,
                    'Acc_Srednia': folds['test_acc'].mean(), 'Acc_Std': folds['test_acc'].std(),
                    'Prec_Srednia': folds['test_prec'].mean(), 'Prec_Std': folds['test_prec'].std(),
                    'Rec_Srednia': folds['test_rec'].mean(), 'Rec_Std': folds['test_rec'].std(),
                    'F1_Srednia': folds['test_f1'].mean(), 'F1_Std': folds['test_f1'].std()
                }
                if domain == "Ptaki" and clean_name in BIRDS_ORDER:
                    birds_rows.append(res_dict)
                elif domain == "Samoloty" and clean_name in PLANES_ORDER:
                    planes_rows.append(res_dict)

        if birds_rows:
            df_b = pd.DataFrame(birds_rows)
            df_b['sort_idx'] = df_b['Eksperyment'].apply(lambda x: BIRDS_ORDER.index(x) if x in BIRDS_ORDER else 99)
            df_b = df_b.sort_values('sort_idx').drop(columns=['sort_idx']).round(2)
            df_b.to_excel(self.writer, sheet_name="5_1_Metryki_Ptaki", index=False)

        if planes_rows:
            df_p = pd.DataFrame(planes_rows)
            df_p['sort_idx'] = df_p['Eksperyment'].apply(lambda x: PLANES_ORDER.index(x) if x in PLANES_ORDER else 99)
            df_p = df_p.sort_values('sort_idx').drop(columns=['sort_idx']).round(2)
            df_p.to_excel(self.writer, sheet_name="5_1_Metryki_Samoloty", index=False)

    # --- 5.3: EFEKTYWNOŚĆ I KOSZT (ZACHOWANA KOLEJNOŚĆ PLANU) ---
    def extract_section_5_3(self):
        print("Przygotowanie danych dla sekcji 5.3 (Efektywność)...")
        eff_tests = self.df[~self.df['folder_group'].str.contains('tuning', case=False)]

        rows_eff = []
        for _, row in eff_tests.drop_duplicates('subfolder').iterrows():
            raw = self.get_raw_data(row)
            if raw is not None:
                raw['train_time_sec'] = pd.to_numeric(raw['train_time_sec'], errors='coerce')

                summary_rows = raw[raw['type'] == 'SUMMARY'].copy()
                test_accs = pd.to_numeric(summary_rows['test_acc'], errors='coerce')
                test_accs = test_accs.apply(lambda x: x * 100 if pd.notna(x) and x <= 1.0 else x)

                epoch_rows = raw[raw['type'] == 'EPOCH']
                if not epoch_rows.empty:
                    fold_times = epoch_rows.groupby('fold')['train_time_sec'].sum()
                    mean_time = fold_times.mean()
                    std_time = fold_times.std() if len(fold_times) > 1 else 0.0
                else:
                    mean_time = np.nan
                    std_time = 0.0

                domain, clean_name = self.classify_row(row)

                rows_eff.append({
                    'Zbiór danych': domain,
                    'Eksperyment': clean_name,
                    'Średnie Accuracy (%)': test_accs.mean(),
                    'Odchylenie Accuracy (std)': test_accs.std() if len(test_accs) > 1 else 0.0,
                    'Średni czas uczenia folda (s)': mean_time,
                    'Odchylenie czasu (std)': std_time,
                    'Czas inferencji (ms)': pd.to_numeric(summary_rows['inference_ms'], errors='coerce').mean(),
                    'Zużycie VRAM (MB)': pd.to_numeric(summary_rows['vram_mb'], errors='coerce').mean(),
                    'Rozmiar modelu (MB)': pd.to_numeric(summary_rows['model_size_mb'], errors='coerce').mean()
                })

        if rows_eff:
            res_df = pd.DataFrame(rows_eff)

            def get_sort_key(r):
                if r['Zbiór danych'] == 'Ptaki' and r['Eksperyment'] in BIRDS_ORDER:
                    return 0, BIRDS_ORDER.index(r['Eksperyment'])
                elif r['Zbiór danych'] == 'Samoloty' and r['Eksperyment'] in PLANES_ORDER:
                    return 1, PLANES_ORDER.index(r['Eksperyment'])
                return 2, 99

            res_df['sort_idx'] = res_df.apply(get_sort_key, axis=1)
            res_df = res_df.sort_values('sort_idx').drop(columns=['sort_idx']).round(2)
            res_df.to_excel(self.writer, sheet_name="5_3_Efektywnosc_Koszt", index=False)

    # --- 5.4: MATRYCA POD BOX PLOTY (ZACHOWANA KOLEJNOŚĆ KOLUMN) ---
    def extract_section_5_4_boxplots(self):
        print("Przygotowanie pełnej macierzy dla Box Plotów (5.4)...")
        all_main_tests = self.df[~self.df['folder_group'].str.contains('tuning', case=False)]

        matrix_data = {'Fold': [1, 2, 3, 4, 5]}
        columns_to_add = []

        for _, row in all_main_tests.drop_duplicates('subfolder').iterrows():
            raw = self.get_raw_data(row)
            if raw is not None:
                summary_rows = raw[raw['type'] == 'SUMMARY'].sort_values('fold').copy()
                test_accs = pd.to_numeric(summary_rows['test_acc'], errors='coerce')
                test_accs = test_accs.apply(lambda x: x * 100 if pd.notna(x) and x <= 1.0 else x)

                vals = test_accs.values
                if len(vals) < 5:
                    vals = np.pad(vals, (0, 5 - len(vals)), mode='constant', constant_values=np.nan)
                elif len(vals) > 5:
                    vals = vals[:5]

                domain, clean_name = self.classify_row(row)
                full_col_name = f"{domain} - {clean_name}"

                if domain == 'Ptaki' and clean_name in BIRDS_ORDER:
                    sort_key = (0, BIRDS_ORDER.index(clean_name))
                elif domain == 'Samoloty' and clean_name in PLANES_ORDER:
                    sort_key = (1, PLANES_ORDER.index(clean_name))
                else:
                    sort_key = (2, 99)

                columns_to_add.append((sort_key, full_col_name, vals))

        columns_to_add.sort(key=lambda x: x[0])
        for _, col_name, vals in columns_to_add:
            matrix_data[col_name] = vals

        if len(matrix_data) > 1:
            res_df = pd.DataFrame(matrix_data).round(2)
            res_df.to_excel(self.writer, sheet_name="5_4_Stabilnosc_BoxPlot", index=False)

    # --- 5.4: ZBIORCZY TUNING PARAMETRÓW (BEZBŁĘDNE WYKRYWANIE I DEKODOWANIE) ---
    def extract_section_5_4_tuning(self):
        print("Przygotowanie danych z optymalizacji tuningu (5.4)...")
        tuning_groups = self.df[self.df['folder_group'].str.contains('tuning', case=False)].copy()

        def extract_param_and_value(subfolder_str):
            s = re.sub(r'^Tune_', '', subfolder_str, flags=re.IGNORECASE)
            s = re.sub(r'_Trial_\d+$', '', s, flags=re.IGNORECASE)
            s = re.sub(r'_Trial_\\d+$', '', s, flags=re.IGNORECASE)
            s = re.sub(r'^(Ptaki|Samoloty|CUB|Aircraft|FGVC|Full)_', '', s, flags=re.IGNORECASE)

            s_lower = s.lower()

            # Parametry złożone
            if 'width' in s_lower and 'depth' in s_lower:
                val = s.replace('head_width_', '').replace('head_depth_', '').replace('width_', '').replace('depth_',
                                                                                                            '')
                return 'Struktura głowy (Szerokość / Głębokość)', val
            if 'dropout' in s_lower and ('weight_decay' in s_lower or 'wd' in s_lower):
                val = s.replace('head_dropout_', '').replace('weight_decay_', '').replace('dropout_', '').replace('wd_',
                                                                                                                  '')
                return 'Regularyzacja (Dropout / Weight Decay)', val
            if 'reg_dr' in s_lower:
                val = s.replace('Reg_Dr_WD_', '').replace('reg_dr_wd_', '')
                return 'Regularyzacja (Dropout / Weight Decay)', val
            if 'head_w' in s_lower:
                val = s.replace('Head_W_D_', '').replace('head_w_d_', '')
                return 'Struktura głowy (Szerokość / Głębokość)', val

            # Parametry pojedyncze
            param_keywords = {
                'batch_size': 'Rozmiar paczki (Batch Size)',
                'batchsize': 'Rozmiar paczki (Batch Size)',
                'learning_rate': 'Współczynnik uczenia (Learning Rate)',
                'learningrate': 'Współczynnik uczenia (Learning Rate)',
                'lr': 'Współczynnik uczenia (Learning Rate)',
                'unfreeze_blocks': 'Liczba odmrożonych bloków',
                'unfreeze': 'Liczba odmrożonych bloków',
                'pooling_mode': 'Metoda agregacji (Pooling)',
                'pooling': 'Metoda agregacji (Pooling)',
                'head_dropout': 'Współczynnik Dropout',
                'dropout': 'Współczynnik Dropout',
                'weight_decay': 'Weight Decay',
                'optimizer': 'Optymalizator (Optimizer)',
                'epochs': 'Liczba epok'
            }

            for kw, polish_label in param_keywords.items():
                if s_lower.startswith(kw):
                    val = s[len(kw):].lstrip('_')
                    return polish_label, val

            parts = s.rsplit('_', 1)
            if len(parts) == 2:
                p_name = parts[0]
                p_val = parts[1]
                p_name_clean = re.sub(r'^(Ptaki|Samoloty|CUB|Aircraft|FGVC|Full)_', '', p_name, flags=re.IGNORECASE)
                lbl = param_keywords.get(p_name_clean.lower(), p_name_clean)
                return lbl, p_val

            return "Inne", s

        rows_tuning = []
        for _, row in tuning_groups.iterrows():
            param_label, val_str = extract_param_and_value(str(row['subfolder']))
            val_str_clean = VALUE_LABELS.get(val_str, val_str)

            full_context = f"{row.get('folder_group', '')} {row.get('subfolder', '')} {row.get('test_name', '')}".lower()
            if any(w in full_context for w in ['ptaki', 'cub', 'bird']):
                domain = "Ptaki"
            elif any(w in full_context for w in ['samoloty', 'aircraft', 'fgvc', 'plane']):
                domain = "Samoloty"
            else:
                domain = "Inne"

            rows_tuning.append({
                'Zbiór danych': domain,
                'Badany Parametr': param_label,
                'Wariant/Wartość': val_str_clean,
                'Test Accuracy (%)': (pd.to_numeric(row['test_acc'], errors='coerce') * 100)
            })

        if rows_tuning:
            res_df = pd.DataFrame(rows_tuning).round(2)
            res_df = res_df.sort_values(by=['Zbiór danych', 'Badany Parametr', 'Wariant/Wartość']).reset_index(
                drop=True)
            res_df.to_excel(self.writer, sheet_name="5_4_Tuning_Hiperparametry", index=False)

    # --- 5.4: KRZYWE UCZENIA (SPÓJNE NAZWY Z RESZTĄ ARKUSZY) ---
    def extract_section_5_4_curves(self):
        print("Przygotowanie danych pod krzywe uczenia...")
        all_main_tests = self.df[~self.df['folder_group'].str.contains('tuning', case=False)]

        all_curves = []
        for _, row in all_main_tests.drop_duplicates('subfolder').iterrows():
            raw = self.get_raw_data(row)
            if raw is not None:
                epochs = raw[raw['type'] == 'EPOCH'].copy()
                if epochs.empty: continue

                epochs['epoch'] = pd.to_numeric(epochs['epoch'], errors='coerce')
                epochs['train_acc'] = pd.to_numeric(epochs['train_acc'], errors='coerce') * 100
                epochs['val_acc'] = pd.to_numeric(epochs['val_acc'], errors='coerce') * 100
                epochs['train_loss'] = pd.to_numeric(epochs['train_loss'], errors='coerce')
                epochs['val_loss'] = pd.to_numeric(epochs['val_loss'], errors='coerce')

                stats = epochs.groupby('epoch').agg({
                    'train_acc': ['mean', 'std'], 'val_acc': ['mean', 'std'],
                    'train_loss': ['mean', 'std'], 'val_loss': ['mean', 'std']
                }).reset_index()

                stats.columns = ['Epoka', 'Train_Acc_Mean', 'Train_Acc_Std', 'Val_Acc_Mean', 'Val_Acc_Std',
                                 'Train_Loss_Mean', 'Train_Loss_Std', 'Val_Loss_Mean', 'Val_Loss_Std']

                domain, clean_name = self.classify_row(row)
                stats['Eksperyment'] = f"{domain} - {clean_name}"
                all_curves.append(stats)

        if all_curves:
            res_df = pd.concat(all_curves).round(2)
            res_df.to_excel(self.writer, sheet_name="5_4_Krzywe_Uczenia_Dane", index=False)

    def save_and_close(self):
        self.writer.close()
        print(f"\nSUKCES: Wszystkie tabele robocze zostały zapisane do pliku: '{OUTPUT_FILE}'")


if __name__ == "__main__":
    if os.path.exists("master_manifest.csv"):
        extractor = ExcelDataExtractor("master_manifest.csv")
        extractor.extract_section_5_1()
        extractor.extract_section_5_3()
        extractor.extract_section_5_4_boxplots()
        extractor.extract_section_5_4_tuning()
        extractor.extract_section_5_4_curves()
        extractor.save_and_close()
    else:
        print("BŁĄD: Brak pliku master_manifest.csv w tym katalogu!")