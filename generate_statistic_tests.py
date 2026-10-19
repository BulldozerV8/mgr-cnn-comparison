import os
import pandas as pd
import scipy.stats as stats

# 1. Dane wejściowe (Accuracy dla 5 foldów)
data = {
    "Zbiór": [
        "Ptaki 20 klas (448px)", "Ptaki 20 klas (224px)", "Ptaki wodne 20 klas", "Ptaki 200 klas",
        "Samoloty 20 klas (448px)", "Samoloty 20 klas (224px)", "Samoloty drugie 20 klas", "Samoloty 101 klas"
    ],
    "EffNetB0": [
        [94.89, 90.34, 91.48, 94.32, 92.05],
        [84.09, 84.09, 85.23, 82.39, 81.82],
        [80.00, 79.43, 77.14, 80.00, 85.14],
        [82.09, 80.34, 81.24, 81.81, 79.83],
        [90.65, 92.24, 91.58, 90.93, 89.99],
        [86.34, 87.46, 88.21, 87.93, 87.37],
        [92.35, 92.06, 93.13, 93.03, 92.84],
        [86.13, 86.09, 85.28, 84.90, 85.72]
    ],
    "Custom_Model": [
        [79.55, 81.82, 85.23, 84.66, 81.25],
        [81.25, 79.55, 76.70, 81.82, 82.95],
        [65.71, 62.86, 62.86, 69.71, 69.14],
        [62.71, 63.56, 62.66, 63.11, 64.01],
        [66.60, 66.14, 64.83, 66.51, 65.29],
        [63.70, 63.70, 63.05, 66.32, 65.29],
        [77.15, 73.67, 75.22, 74.06, 74.93],
        [32.41, 33.68, 33.23, 30.93, 33.75]
    ]
}

results = []

# 2. Obliczenia statystyczne
for i in range(len(data["Zbiór"])):
    dataset = data["Zbiór"][i]
    eff = data["EffNetB0"][i]
    custom = data["Custom_Model"][i]

    # Różnice między modelami
    diff = [e - c for e, c in zip(eff, custom)]

    # Test normalności (Shapiro-Wilk)
    stat_shapiro, p_shapiro = stats.shapiro(diff)

    # Dobór testu i obliczenie p-value
    if p_shapiro > 0.05:
        test_name = "t-Studenta (próby zależne)"
        stat_test, p_val = stats.ttest_rel(eff, custom)
    else:
        test_name = "Wilcoxona (znaki rangowane)"
        stat_test, p_val = stats.wilcoxon(eff, custom)

    # Interpretacja
    is_significant = "TAK" if p_val < 0.05 else "NIE"

    results.append({
        "Zbiór danych": dataset,
        "Śr. EffNetB0": round(sum(eff) / 5, 2),
        "Śr. Custom": round(sum(custom) / 5, 2),
        "Rozkład normalny (Shapiro p>0.05)": "TAK" if p_shapiro > 0.05 else "NIE",
        "Użyty test statystyczny": test_name,
        "Wartość p-value": f"{p_val:.4e}",
        "Istotność statystyczna (p < 0.05)": is_significant
    })

# 3. Zapis do Excela
df = pd.DataFrame(results)
out_dir = "Thesis_Outputs"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "Testy_Statystyczne_Modele.xlsx")

df.to_excel(out_path, index=False)
print(f"Gotowe! Plik zapisano w: {out_path}")