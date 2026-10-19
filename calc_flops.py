import torch
from thop import profile
import pandas as pd
import os

# Zmień ścieżkę importu na taką, z jakiej korzystasz w swoim projekcie
from models.cnn_base import CNN

# LISTA KONFIGURACJI DLA WSZYSTKICH OSTATECZNYCH WERSJI MODELI
configs = [
    # ================= EFFICIENTNET-B0 =================
    {
        "model_base": "EfficientNet-B0", "dataset_name": "Ptaki 20 klas (448px)",
        "type": "efficientnet_b0", "num_classes": 20, "res": 448,
        "head_w": 256, "head_d": 2, "pooling": "avg"
    },
    {
        "model_base": "EfficientNet-B0", "dataset_name": "Ptaki 20 klas (224px)",
        "type": "efficientnet_b0", "num_classes": 20, "res": 224,
        "head_w": 256, "head_d": 2, "pooling": "avg"
    },
    {
        "model_base": "EfficientNet-B0", "dataset_name": "Ptaki 200 klas",
        "type": "efficientnet_b0", "num_classes": 200, "res": 448,
        "head_w": 256, "head_d": 2, "pooling": "avg"
    },
    {
        "model_base": "EfficientNet-B0", "dataset_name": "Samoloty 20 klas (448px)",
        "type": "efficientnet_b0", "num_classes": 20, "res": 448,
        "head_w": 512, "head_d": 3, "pooling": "avg"
    },
    {
        "model_base": "EfficientNet-B0", "dataset_name": "Samoloty 20 klas (224px)",
        "type": "efficientnet_b0", "num_classes": 20, "res": 224,
        "head_w": 512, "head_d": 3, "pooling": "avg"
    },
    {
        "model_base": "EfficientNet-B0", "dataset_name": "Samoloty 101 klas",
        "type": "efficientnet_b0", "num_classes": 101, "res": 448,
        "head_w": 512, "head_d": 3, "pooling": "avg"
    },

    # ================= AUTORSKA SIEĆ =================
    {
        "model_base": "MyCustomCNN", "dataset_name": "Ptaki 20 klas (448px)",
        "type": "my_custom_cnn", "num_classes": 20, "res": 448,
        "head_w": 256, "head_d": 3, "pooling": "max",
        "base_ch": 32, "blocks": [3, 3, 4], "se": True, "depthwise": True, "kernel": 5, "expansion": 4
    },
    {
        "model_base": "MyCustomCNN", "dataset_name": "Ptaki 20 klas (224px)",
        "type": "my_custom_cnn", "num_classes": 20, "res": 224,
        "head_w": 256, "head_d": 3, "pooling": "max",
        "base_ch": 32, "blocks": [3, 3, 4], "se": True, "depthwise": True, "kernel": 5, "expansion": 4
    },
    {
        "model_base": "MyCustomCNN", "dataset_name": "Ptaki 200 klas",
        "type": "my_custom_cnn", "num_classes": 200, "res": 448,
        "head_w": 256, "head_d": 3, "pooling": "max",
        "base_ch": 32, "blocks": [3, 3, 4], "se": True, "depthwise": True, "kernel": 5, "expansion": 4
    },
    {
        "model_base": "MyCustomCNN", "dataset_name": "Samoloty 20 klas (448px)",
        "type": "my_custom_cnn", "num_classes": 20, "res": 448,
        "head_w": 256, "head_d": 3, "pooling": "avg",
        "base_ch": 32, "blocks": [3, 3, 4], "se": True, "depthwise": True, "kernel": 5, "expansion": 4
    },
    {
        "model_base": "MyCustomCNN", "dataset_name": "Samoloty 20 klas (224px)",
        "type": "my_custom_cnn", "num_classes": 20, "res": 224,
        "head_w": 256, "head_d": 3, "pooling": "avg",
        "base_ch": 32, "blocks": [3, 3, 4], "se": True, "depthwise": True, "kernel": 5, "expansion": 4
    },
    {
        "model_base": "MyCustomCNN", "dataset_name": "Samoloty 101 klas",
        "type": "my_custom_cnn", "num_classes": 101, "res": 448,
        "head_w": 256, "head_d": 3, "pooling": "avg",
        "base_ch": 32, "blocks": [3, 3, 4], "se": True, "depthwise": True, "kernel": 5, "expansion": 4
    },
]

results = []
print("Rozpoczynam obliczanie FLOPs i Parametrów...\n")

for c in configs:
    # 1. Zbuduj model
    if c["type"] == "efficientnet_b0":
        model = CNN(
            num_classes=c["num_classes"],
            model_name="efficientnet_b0",
            head_width=c["head_w"],
            head_depth=c["head_d"],
            pooling_mode=c["pooling"]
        )
    else:
        model = CNN(
            num_classes=c["num_classes"],
            model_name="my_custom_cnn",
            custom_base_channels=c["base_ch"],
            custom_block_repeats=c["blocks"],
            use_se_block=c["se"],
            use_depthwise=c["depthwise"],
            kernel_size=c["kernel"],
            expansion_ratio=c["expansion"],
            head_width=c["head_w"],
            head_depth=c["head_d"],
            pooling_mode=c["pooling"]
        )

    # 2. Utwórz wejściowy tensor symulujący jedno zdjęcie
    dummy_input = torch.randn(1, 3, c["res"], c["res"])

    # 3. Profilowanie za pomocą THOP (tylko dla FLOPs)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

    # 4. Natywne, bezbłędne liczenie parametrów przez PyTorch (zgodne z Twoimi logami 1:1)
    params = sum(p.numel() for p in model.parameters())

    # 5. Zapisz wyniki
    results.append({
        "Architektura bazowa": c["model_base"],
        "Zbiór (Eksperyment)": c["dataset_name"],
        "FLOPs (Giga)": round(flops / 1e9, 3),
        "Parametry (Miliony)": round(params / 1e6, 3)
    })

    print(f"Obliczono: {c['model_base']} - {c['dataset_name']}")

# 6. Eksport do Excela
output_path = os.path.join("Thesis_Outputs", "Koszty_Obliczeniowe_Modele.xlsx")
os.makedirs("Thesis_Outputs", exist_ok=True)

df = pd.DataFrame(results)
df.to_excel(output_path, index=False, engine='openpyxl')
print(f"\nZakończono sukcesem! Tabela zapisana w: {output_path}")