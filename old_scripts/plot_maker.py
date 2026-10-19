import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =================================================================
# KONFIGURACJA - DOSTOSOWANA DO POLSKIEJ PRACY MAGISTERSKIEJ
# =================================================================
FILE_SETTINGS = {
    "input_file": "results/FINAL_Samoloty_448px_CV5/FINAL_Samoloty_448px_CV5_results.csv",
    "output_dir": "results_plots/Analiza_Samoloty_V1",
    "plot_name": "Krzywa_uczenia_dokladnosc_samoloty.png",  # Nazwa pliku po polsku
}

PLOT_SETTINGS = {
    #"title": "Krzywa uczenia - Dokładność klasyfikacji",  # Polski tytuł
    "x_axis": "epoch",
    "y_axes": ["train_acc", "val_acc"],
    "y_label": "Dokładność (%)",  # Oś Y jest git
    "x_label": "Epoka",  # Krótko i profesjonalnie
    "filter_col": "type",
    "filter_val": "EPOCH",
}

# MAPOWANIE LEGENDY - Tutaj definiujesz ładne nazwy
LEGEND_MAP = {
    "train_acc": "Zbiór treningowy",
    "val_acc": "Zbiór walidacyjny",
    "train_loss": "Strata (trening)",
    "val_loss": "Strata (walidacja)"
}

STYLE = {
    "dpi": 300,
    "figsize": (10, 6),
    "palette": "bright",
}


# =================================================================

def run_research_plotter():
    os.makedirs(FILE_SETTINGS["output_dir"], exist_ok=True)
    df = pd.read_csv(FILE_SETTINGS["input_file"])

    df = df[df[PLOT_SETTINGS["filter_col"]] == PLOT_SETTINGS["filter_val"]].copy()

    cols = [PLOT_SETTINGS["x_axis"]] + PLOT_SETTINGS["y_axes"]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    # Przeliczenie na procenty, jeśli dane są w formacie 0.0 - 1.0
    for y_col in PLOT_SETTINGS["y_axes"]:
        if df[y_col].max() <= 1.0:
            df[y_col] = df[y_col] * 100

    fig, ax = plt.subplots(figsize=STYLE["figsize"], dpi=STYLE["dpi"])
    sns.set_theme(style="whitegrid")

    for y_col in PLOT_SETTINGS["y_axes"]:
        # Pobieramy ładną nazwę ze słownika, jeśli nie ma - zostaje stara
        nice_name = LEGEND_MAP.get(y_col, y_col)

        sns.lineplot(
            data=df,
            x=PLOT_SETTINGS["x_axis"],
            y=y_col,
            ax=ax,
            label=nice_name,  # Używamy zmapowanej nazwy
            marker='o',
            markersize=5
        )

    #ax.set_title(PLOT_SETTINGS["title"], fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(PLOT_SETTINGS["x_label"])
    ax.set_ylabel(PLOT_SETTINGS["y_label"])
    ax.legend(frameon=True, loc='best')

    # Ustawienie osi X tak, by pokazywała liczby całkowite (epoki)
    ax.set_xticks(range(int(df[PLOT_SETTINGS["x_axis"]].min()), int(df[PLOT_SETTINGS["x_axis"]].max()) + 1))

    save_path = os.path.join(FILE_SETTINGS["output_dir"], FILE_SETTINGS["plot_name"])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Wykres zapisany: {save_path}")


if __name__ == "__main__":
    run_research_plotter()