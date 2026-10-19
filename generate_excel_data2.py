import os
import pandas as pd
import numpy as np
import re

OUTPUT_FILE = "Custom_Data_For_Excel.xlsx"

TARGET_FOLDERS = [
    # --- FOLDERY GŁÓWNE ---
    "Custom_Final_Ptaki_448px",
    "Custom_Final_Ptaki_224px",
    "Custom_Final_Ptaki_wodne",
    "Custom_Final_Ptaki_200_klas",
    "Custom_Final_Ptaki_pogoda",
    "Custom_Final_Samoloty_448px",
    "Custom_Final_Samoloty_224px",
    "Custom_Final_Samoloty_drugie",
    "Custom_Final_Samoloty_200_klas",
    "Custom_Final_Samoloty_pogoda",
    # --- FOLDERY TUNINGU ---
    "1_Final_tuning_ptaki_wlasny_model",
    "1_Final_tuning_samoloty_wlasny_model"
]

# Sztywna kolejność sekcji z planu
BIRDS_ORDER = [
    'Ptaki 20 klas (448px)',
    'Ptaki 20 klas (224px)',
    'Ptaki wodne 20 klas',
    'Ptaki 200 klas',
    'Ptaki pogoda 20 klas',
]

PLANES_ORDER = [
    'Samoloty 20 klas (448px)',
    'Samoloty 20 klas (224px)',
    'Samoloty drugie 20 klas',
    'Samoloty 101 klas',
    'Samoloty pogoda 20 klas',
]

VALUE_LABELS = {
    'avg': 'Średnia (Avg)', 'max': 'Maksymalna (Max)',
    'pooling_mode_avg': 'Średnia (Avg)', 'pooling_mode_max': 'Maksymalna (Max)',
}

# MAPOWANIE: Nazwa folderu -> (Domena, Nazwa Eksperymentu)
FOLDER_TO_NAME = {
    "Custom_Final_Ptaki_448px": ("Ptaki", "Ptaki 20 klas (448px)"),
    "Custom_Final_Ptaki_224px": ("Ptaki", "Ptaki 20 klas (224px)"),
    "Custom_Final_Ptaki_wodne": ("Ptaki", "Ptaki wodne 20 klas"),
    "Custom_Final_Ptaki_200_klas": ("Ptaki", "Ptaki 200 klas"),
    "Custom_Final_Ptaki_pogoda": ("Ptaki", "Ptaki pogoda 20 klas"),
    "Custom_Final_Samoloty_448px": ("Samoloty", "Samoloty 20 klas (448px)"),
    "Custom_Final_Samoloty_224px": ("Samoloty", "Samoloty 20 klas (224px)"),
    "Custom_Final_Samoloty_drugie": ("Samoloty", "Samoloty drugie 20 klas"),
    "Custom_Final_Samoloty_200_klas": ("Samoloty", "Samoloty 101 klas"),
    "Custom_Final_Samoloty_pogoda": ("Samoloty", "Samoloty pogoda 20 klas"),
}


def format_ms(mean_val, std_val):
    if pd.isna(mean_val): return "-"
    if pd.isna(std_val) or std_val == 0.0: return f"{mean_val:.2f} ± 0.00"
    return f"{mean_val:.2f} ± {std_val:.2f}"


def merge_dfs_with_gap(df1, df2, cols, add_header=False):
    """Funkcja łącząca dwa df-y dodająca pusty wiersz i opcjonalnie podwójny nagłówek"""
    if df1.empty and df2.empty: return pd.DataFrame(columns=cols)
    if df1.empty: return df2
    if df2.empty: return df1

    empty_row = pd.DataFrame([[np.nan] * len(cols)], columns=cols)
    parts = [df1, empty_row]
    if add_header:
        header_row = pd.DataFrame([cols], columns=cols)
        parts.append(header_row)
    parts.append(df2)
    return pd.concat(parts, ignore_index=True)


class ExcelDataExtractor:
    def __init__(self, target_folders):
        data = []
        for folder in target_folders:
            folder_path = os.path.join("results", folder)
            if not os.path.exists(folder_path):
                continue
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".csv") and "_results" in file:
                        data.append({
                            'subfolder': os.path.relpath(root, "results"),
                            'file_source': file,
                            'test_name': folder
                        })
        self.df = pd.DataFrame(data)
        self.writer = pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl')

    def classify_row(self, row):
        folder = row.get('test_name', '')
        if folder in FOLDER_TO_NAME:
            return FOLDER_TO_NAME[folder]
        return "Tuning", row['subfolder']

    def get_raw_data(self, row):
        path = os.path.join("results", row['subfolder'], row['file_source'])
        return pd.read_csv(path) if os.path.exists(path) else None

    # --- 5.1 SUROWE DANE ---
    def extract_section_5_1_raw(self):
        print("Przygotowanie surowych danych dla sekcji 5.1...")
        cols = ['Zbiór', 'Accuracy Średnia', 'Acc_Std', 'Precision Średnia', 'Prec_Std',
                'Recall Średnia', 'Rec_Std', 'F1-score Średnia', 'F1_Std',
                'Czas (w minutach)', 'Czas std (w minutach)']

        b_rows, p_rows = [], []
        main_tests = self.df[self.df['test_name'].isin(FOLDER_TO_NAME.keys())].copy()

        for _, row in main_tests.drop_duplicates('test_name').iterrows():
            domain, clean_name = self.classify_row(row)
            raw = self.get_raw_data(row)
            if raw is not None:
                folds = raw[raw['type'] == 'SUMMARY'].copy()
                for m in ['test_acc', 'test_prec', 'test_rec', 'test_f1']:
                    folds[m] = pd.to_numeric(folds[m], errors='coerce').apply(
                        lambda x: x * 100 if pd.notna(x) and x <= 1.0 else x)

                epoch_rows = raw[raw['type'] == 'EPOCH'].copy()
                if not epoch_rows.empty:
                    epoch_rows['train_time_sec'] = pd.to_numeric(epoch_rows['train_time_sec'], errors='coerce')
                    fold_times_min = epoch_rows.groupby('fold')['train_time_sec'].sum() / 60.0
                    mean_t, std_t = fold_times_min.mean(), (fold_times_min.std() if len(fold_times_min) > 1 else 0.0)
                else:
                    mean_t, std_t = 0.0, 0.0

                res = {
                    'Zbiór': clean_name,
                    'Accuracy Średnia': folds['test_acc'].mean(),
                    'Acc_Std': folds['test_acc'].std() if len(folds) > 1 else 0.0,
                    'Precision Średnia': folds['test_prec'].mean(),
                    'Prec_Std': folds['test_prec'].std() if len(folds) > 1 else 0.0,
                    'Recall Średnia': folds['test_rec'].mean(),
                    'Rec_Std': folds['test_rec'].std() if len(folds) > 1 else 0.0,
                    'F1-score Średnia': folds['test_f1'].mean(),
                    'F1_Std': folds['test_f1'].std() if len(folds) > 1 else 0.0,
                    'Czas (w minutach)': mean_t, 'Czas std (w minutach)': std_t
                }
                if domain == "Ptaki":
                    b_rows.append(res)
                else:
                    p_rows.append(res)

        df_b = pd.DataFrame(b_rows, columns=cols)
        df_p = pd.DataFrame(p_rows, columns=cols)

        if not df_b.empty:
            df_b['idx'] = df_b['Zbiór'].apply(lambda x: BIRDS_ORDER.index(x) if x in BIRDS_ORDER else 99)
            df_b = df_b.sort_values('idx').drop(columns=['idx'])
        if not df_p.empty:
            df_p['idx'] = df_p['Zbiór'].apply(lambda x: PLANES_ORDER.index(x) if x in PLANES_ORDER else 99)
            df_p = df_p.sort_values('idx').drop(columns=['idx'])

        final_df = merge_dfs_with_gap(df_b, df_p, cols, add_header=False)
        final_df.to_excel(self.writer, sheet_name="5_1_Surowe_Dane", index=False)

    # --- 5.1 GOTOWE TABELKI (Z ± I PODWÓJNYM NAGŁÓWKIEM) ---
    def extract_section_5_1_formatted(self):
        print("Przygotowanie gotowych tabelek dla sekcji 5.1...")
        cols = ['Zbiór', 'Accurancy (%)', 'Precision (%)', 'Recall (%)', 'F1-score (%)', 'Czas uczenia (min)']
        b_rows, p_rows = [], []
        main_tests = self.df[self.df['test_name'].isin(FOLDER_TO_NAME.keys())].copy()

        for _, row in main_tests.drop_duplicates('test_name').iterrows():
            domain, clean_name = self.classify_row(row)
            raw = self.get_raw_data(row)
            if raw is not None:
                folds = raw[raw['type'] == 'SUMMARY'].copy()
                for m in ['test_acc', 'test_prec', 'test_rec', 'test_f1']:
                    folds[m] = pd.to_numeric(folds[m], errors='coerce').apply(
                        lambda x: x * 100 if pd.notna(x) and x <= 1.0 else x)

                epoch_rows = raw[raw['type'] == 'EPOCH'].copy()
                if not epoch_rows.empty:
                    epoch_rows['train_time_sec'] = pd.to_numeric(epoch_rows['train_time_sec'], errors='coerce')
                    fold_times_min = epoch_rows.groupby('fold')['train_time_sec'].sum() / 60.0
                    mean_t, std_t = fold_times_min.mean(), (fold_times_min.std() if len(fold_times_min) > 1 else 0.0)
                else:
                    mean_t, std_t = 0.0, 0.0

                res = {
                    'Zbiór': clean_name,
                    'Accurancy (%)': format_ms(folds['test_acc'].mean(),
                                               folds['test_acc'].std() if len(folds) > 1 else 0.0),
                    'Precision (%)': format_ms(folds['test_prec'].mean(),
                                               folds['test_prec'].std() if len(folds) > 1 else 0.0),
                    'Recall (%)': format_ms(folds['test_rec'].mean(),
                                            folds['test_rec'].std() if len(folds) > 1 else 0.0),
                    'F1-score (%)': format_ms(folds['test_f1'].mean(),
                                              folds['test_f1'].std() if len(folds) > 1 else 0.0),
                    'Czas uczenia (min)': format_ms(mean_t, std_t)
                }
                if domain == "Ptaki":
                    b_rows.append(res)
                else:
                    p_rows.append(res)

        df_b = pd.DataFrame(b_rows, columns=cols)
        df_p = pd.DataFrame(p_rows, columns=cols)

        if not df_b.empty:
            df_b['idx'] = df_b['Zbiór'].apply(lambda x: BIRDS_ORDER.index(x) if x in BIRDS_ORDER else 99)
            df_b = df_b.sort_values('idx').drop(columns=['idx'])
        if not df_p.empty:
            df_p['idx'] = df_p['Zbiór'].apply(lambda x: PLANES_ORDER.index(x) if x in PLANES_ORDER else 99)
            df_p = df_p.sort_values('idx').drop(columns=['idx'])

        final_df = merge_dfs_with_gap(df_b, df_p, cols, add_header=True)
        final_df.to_excel(self.writer, sheet_name="5_1_Gotowe_Tabelki", index=False)

    # --- 5.3 SUROWE DANE ---
    def extract_section_5_3_raw(self):
        print("Przygotowanie surowych danych dla sekcji 5.3...")
        cols = ['Zbiór', 'Średnie Accuracy (%)', 'Odchylenie Accuracy (std)',
                'Średni czas uczenia folda (min)', 'Odchylenie czasu (std) (min)', 'Zużycie VRAM (MB)']
        b_rows, p_rows = [], []
        main_tests = self.df[self.df['test_name'].isin(FOLDER_TO_NAME.keys())].copy()

        for _, row in main_tests.drop_duplicates('test_name').iterrows():
            domain, clean_name = self.classify_row(row)
            raw = self.get_raw_data(row)
            if raw is not None:
                summary = raw[raw['type'] == 'SUMMARY'].copy()
                accs = pd.to_numeric(summary['test_acc'], errors='coerce').apply(
                    lambda x: x * 100 if pd.notna(x) and x <= 1.0 else x)
                vrams = pd.to_numeric(summary['vram_mb'], errors='coerce')

                epoch_rows = raw[raw['type'] == 'EPOCH'].copy()
                if not epoch_rows.empty:
                    epoch_rows['train_time_sec'] = pd.to_numeric(epoch_rows['train_time_sec'], errors='coerce')
                    fold_times_min = epoch_rows.groupby('fold')['train_time_sec'].sum() / 60.0
                    mean_t, std_t = fold_times_min.mean(), (fold_times_min.std() if len(fold_times_min) > 1 else 0.0)
                else:
                    mean_t, std_t = 0.0, 0.0

                res = {
                    'Zbiór': clean_name,
                    'Średnie Accuracy (%)': accs.mean(),
                    'Odchylenie Accuracy (std)': accs.std() if len(accs) > 1 else 0.0,
                    'Średni czas uczenia folda (min)': mean_t, 'Odchylenie czasu (std) (min)': std_t,
                    'Zużycie VRAM (MB)': vrams.mean()
                }
                if domain == "Ptaki":
                    b_rows.append(res)
                else:
                    p_rows.append(res)

        df_b = pd.DataFrame(b_rows, columns=cols)
        df_p = pd.DataFrame(p_rows, columns=cols)

        if not df_b.empty:
            df_b['idx'] = df_b['Zbiór'].apply(lambda x: BIRDS_ORDER.index(x) if x in BIRDS_ORDER else 99)
            df_b = df_b.sort_values('idx').drop(columns=['idx'])
        if not df_p.empty:
            df_p['idx'] = df_p['Zbiór'].apply(lambda x: PLANES_ORDER.index(x) if x in PLANES_ORDER else 99)
            df_p = df_p.sort_values('idx').drop(columns=['idx'])

        final_df = merge_dfs_with_gap(df_b, df_p, cols, add_header=False)
        final_df.to_excel(self.writer, sheet_name="5_3_Efektywnosc_Koszt", index=False)

    # --- 5.3 GOTOWE TABELKI (Z ± I PODWÓJNYM NAGŁÓWKIEM) ---
    def extract_section_5_3_formatted(self):
        print("Przygotowanie gotowych tabelek dla sekcji 5.3...")
        cols = ['Model / Rozdzielczość', 'Średnie Accuracy (%)', 'Średni czas uczenia folda (min)', 'Zużycie VRAM (MB)']
        b_rows, p_rows = [], []
        main_tests = self.df[self.df['test_name'].isin(FOLDER_TO_NAME.keys())].copy()

        for _, row in main_tests.drop_duplicates('test_name').iterrows():
            domain, clean_name = self.classify_row(row)
            raw = self.get_raw_data(row)
            if raw is not None:
                summary = raw[raw['type'] == 'SUMMARY'].copy()
                accs = pd.to_numeric(summary['test_acc'], errors='coerce').apply(
                    lambda x: x * 100 if pd.notna(x) and x <= 1.0 else x)
                vrams = pd.to_numeric(summary['vram_mb'], errors='coerce')

                epoch_rows = raw[raw['type'] == 'EPOCH'].copy()
                if not epoch_rows.empty:
                    epoch_rows['train_time_sec'] = pd.to_numeric(epoch_rows['train_time_sec'], errors='coerce')
                    fold_times_min = epoch_rows.groupby('fold')['train_time_sec'].sum() / 60.0
                    mean_t, std_t = fold_times_min.mean(), (fold_times_min.std() if len(fold_times_min) > 1 else 0.0)
                else:
                    mean_t, std_t = 0.0, 0.0

                res = {
                    'Model / Rozdzielczość': clean_name,
                    'Średnie Accuracy (%)': format_ms(accs.mean(), accs.std() if len(accs) > 1 else 0.0),
                    'Średni czas uczenia folda (min)': format_ms(mean_t, std_t),
                    'Zużycie VRAM (MB)': round(vrams.mean(), 2) if not pd.isna(vrams.mean()) else "-"
                }
                if domain == "Ptaki":
                    b_rows.append(res)
                else:
                    p_rows.append(res)

        df_b = pd.DataFrame(b_rows, columns=cols)
        df_p = pd.DataFrame(p_rows, columns=cols)

        if not df_b.empty:
            df_b['idx'] = df_b['Model / Rozdzielczość'].apply(
                lambda x: BIRDS_ORDER.index(x) if x in BIRDS_ORDER else 99)
            df_b = df_b.sort_values('idx').drop(columns=['idx'])
        if not df_p.empty:
            df_p['idx'] = df_p['Model / Rozdzielczość'].apply(
                lambda x: PLANES_ORDER.index(x) if x in PLANES_ORDER else 99)
            df_p = df_p.sort_values('idx').drop(columns=['idx'])

        final_df = merge_dfs_with_gap(df_b, df_p, cols, add_header=True)
        final_df.to_excel(self.writer, sheet_name="5_3_Gotowe_Tabelki", index=False)

    # --- 5.4 BOXPLOTY ---
    def extract_section_5_4_boxplots(self):
        print("Przygotowanie macierzy dla Box Plotów (5.4)...")
        main_tests = self.df[self.df['test_name'].isin(FOLDER_TO_NAME.keys())].copy()

        matrix_data = {'Fold': [1, 2, 3, 4, 5]}
        b_cols, p_cols = [], []

        for _, row in main_tests.drop_duplicates('test_name').iterrows():
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
                if domain == 'Ptaki':
                    sort_idx = BIRDS_ORDER.index(clean_name) if clean_name in BIRDS_ORDER else 99
                    b_cols.append((sort_idx, clean_name, vals))
                else:
                    sort_idx = PLANES_ORDER.index(clean_name) if clean_name in PLANES_ORDER else 99
                    p_cols.append((sort_idx, clean_name, vals))

        b_cols.sort(key=lambda x: x[0])
        p_cols.sort(key=lambda x: x[0])

        for _, col_name, vals in b_cols + p_cols:
            matrix_data[col_name] = vals

        res_df = pd.DataFrame(matrix_data).round(2)
        res_df.to_excel(self.writer, sheet_name="5_4_Stabilnosc_BoxPlot", index=False)

    # --- 5.4 TUNING HIPERPARAMETRÓW ---
    def extract_section_5_4_tuning(self):
        print("Przygotowanie danych z optymalizacji tuningu (5.4)...")
        tuning_tests = self.df[~self.df['test_name'].isin(FOLDER_TO_NAME.keys())].copy()

        if tuning_tests.empty:
            print("INFO: Brak folderów tuningu! Upewnij się, że są dodane do TARGET_FOLDERS.")
            return

        def extract_param_and_value(subfolder_str):
            # Użycie basename zapobiega błędom, gdy w ścieżce są podfoldery
            base_folder = os.path.basename(subfolder_str)
            s = re.sub(r'^Tune_', '', base_folder, flags=re.IGNORECASE)
            s = re.sub(r'_Trial_\d+$', '', s, flags=re.IGNORECASE)

            s_lower = s.lower()
            if 'width' in s_lower and 'depth' in s_lower:
                val = s.replace('head_width_', '').replace('head_depth_', '').replace('width_', '').replace('depth_', '')
                return 'Struktura głowy (Szerokość / Głębokość)', val
            if 'dropout' in s_lower and ('weight_decay' in s_lower or 'wd' in s_lower):
                val = s.replace('head_dropout_', '').replace('weight_decay_', '').replace('dropout_', '').replace('wd_', '')
                return 'Regularyzacja (Dropout / Weight Decay)', val
            if 'reg_dr' in s_lower:
                val = s.replace('Reg_Dr_WD_', '').replace('reg_dr_wd_', '')
                return 'Regularyzacja (Dropout / Weight Decay)', val
            if 'head_w' in s_lower:
                val = s.replace('Head_W_D_', '').replace('head_w_d_', '')
                return 'Struktura głowy (Szerokość / Głębokość)', val

            # Rozbudowana i ułożona lista słów kluczowych (od najdłuższych)
            param_keywords = {
                'num_samples_per_epoch': 'Liczba próbek na epokę (Samples per Epoch)',
                'custom_base_channels': 'Liczba kanałów bazowych (Base Channels)',
                'custom_block_repeats': 'Liczba powtórzeń bloków (Block Repeats)',
                'unfreeze_blocks': 'Liczba odmrożonych bloków',
                'expansion_ratio': 'Mnożnik rozszerzenia (Expansion Ratio)',
                'drop_path_rate': 'Stochastyczna głębokość (Drop Path Rate)',
                'learning_rate': 'Współczynnik uczenia (Learning Rate)',
                'pooling_mode': 'Metoda agregacji (Pooling)',
                'use_depthwise': 'Sploty separowalne (Depthwise)',
                'weight_decay': 'Weight Decay',
                'head_dropout': 'Współczynnik Dropout',
                'batch_size': 'Rozmiar paczki (Batch Size)',
                'kernel_size': 'Rozmiar jądra splotu (Kernel Size)',
                'base_channels': 'Liczba kanałów bazowych (Base Channels)',
                'block_repeats': 'Liczba powtórzeń bloków (Block Repeats)',
                'depthwise': 'Sploty separowalne (Depthwise)',
                'drop_path': 'Stochastyczna głębokość (Drop Path Rate)',
                'unfreeze': 'Liczba odmrożonych bloków',
                'pooling': 'Metoda agregacji (Pooling)',
                'dropout': 'Współczynnik Dropout',
                'samples': 'Liczba próbek na epokę (Samples per Epoch)',
                'kernel': 'Rozmiar jądra splotu (Kernel Size)',
                'epochs': 'Liczba epok',
                'lr': 'Współczynnik uczenia (Learning Rate)',
            }

            for kw, polish_label in param_keywords.items():
                if s_lower.startswith(kw):
                    val = re.sub(f"^{kw}_?", "", s, flags=re.IGNORECASE)
                    return polish_label, val

            parts = s.rsplit('_', 1)
            if len(parts) == 2:
                p_name_clean = re.sub(r'^(Ptaki|Samoloty|CUB|Aircraft|FGVC|Full)_', '', parts[0], flags=re.IGNORECASE)
                return param_keywords.get(p_name_clean.lower(), p_name_clean), parts[1]

            return "Inne", s

        rows_tuning = []
        for _, row in tuning_tests.iterrows():
            param_label, val_str = extract_param_and_value(str(row['subfolder']))
            val_str_clean = VALUE_LABELS.get(val_str, val_str)

            full_context = f"{row.get('subfolder', '')} {row.get('test_name', '')}".lower()
            domain = "Ptaki" if any(w in full_context for w in ['ptaki', 'cub', 'bird']) else "Samoloty" if any(
                w in full_context for w in ['samoloty', 'aircraft', 'fgvc', 'plane']) else "Inne"

            # Pobranie wartości z wiersza SUMMARY z pliku CSV
            raw = self.get_raw_data(row)
            test_acc_val = 0.0
            if raw is not None:
                summary_rows = raw[raw['type'] == 'SUMMARY'].copy()
                if not summary_rows.empty:
                    accs = pd.to_numeric(summary_rows['test_acc'], errors='coerce')
                    test_acc_val = accs.apply(lambda x: x * 100 if pd.notna(x) and x <= 1.0 else x).mean()

            rows_tuning.append({
                'Zbiór danych': domain,
                'Badany Parametr': param_label,
                'Wariant/Wartość': val_str_clean,
                'Test Accuracy (%)': test_acc_val
            })

        if rows_tuning:
            res_df = pd.DataFrame(rows_tuning).round(2)

            # SZTYWNA KOLEJNOŚĆ WYŚWIETLANIA W EXCELU (zgodna z Twoimi instrukcjami)
            PARAM_ORDER = [
                'Rozmiar paczki (Batch Size)',
                'Współczynnik uczenia (Learning Rate)',
                'Liczba odmrożonych bloków',
                'Liczba kanałów bazowych (Base Channels)',
                'Liczba powtórzeń bloków (Block Repeats)',
                'Mnożnik rozszerzenia (Expansion Ratio)',
                'Sploty separowalne (Depthwise)',
                'Rozmiar jądra splotu (Kernel Size)',
                'Struktura głowy (Szerokość / Głębokość)',
                'Stochastyczna głębokość (Drop Path Rate)',
                'Metoda agregacji (Pooling)',
                'Regularyzacja (Dropout / Weight Decay)',
                'Liczba próbek na epokę (Samples per Epoch)',
                'Współczynnik Dropout',
                'Weight Decay',
                'Liczba epok',
                'Inne'
            ]

            # Tworzenie tymczasowej kolumny sortującej
            res_df['param_idx'] = res_df['Badany Parametr'].apply(
                lambda x: PARAM_ORDER.index(x) if x in PARAM_ORDER else 99
            )

            # Sortowanie według zbioru -> następnie naszej sztywnej listy -> następnie wariantu
            res_df = res_df.sort_values(by=['Zbiór danych', 'param_idx', 'Wariant/Wartość'])

            # Sprzątanie kolumny pomocniczej przed zapisem
            res_df = res_df.drop(columns=['param_idx']).reset_index(drop=True)
            res_df.to_excel(self.writer, sheet_name="5_4_Tuning_Hiperparametry", index=False)

    def save_and_close(self):
        self.writer.close()
        print(f"\nSUKCES: Zapisano do: '{OUTPUT_FILE}'")


if __name__ == "__main__":
    extractor = ExcelDataExtractor(TARGET_FOLDERS)
    extractor.extract_section_5_1_raw()
    extractor.extract_section_5_1_formatted()
    extractor.extract_section_5_3_raw()
    extractor.extract_section_5_3_formatted()
    extractor.extract_section_5_4_boxplots()
    extractor.extract_section_5_4_tuning()
    extractor.save_and_close()