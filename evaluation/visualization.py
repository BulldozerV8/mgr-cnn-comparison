import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Konfiguracja estetyki wykresów zgodnie ze standardami publikacyjnymi
sns.set_theme(style="whitegrid", palette="muted")


class Visualizer:
    """
    Klasa odpowiedzialna za wizualną interpretację wyników eksperymentu.
    Dostarcza narzędzi do generowania krzywych uczenia oraz macierzy pomyłek,
    dbając o wysoką rozdzielczość plików wyjściowych i czytelność danych.
    """

    @staticmethod
    def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, fold, save_path):
        """
        Generuje i archiwizuje krzywe uczenia (Loss oraz Accuracy).
        Pozwala na ocenę dynamiki optymalizacji modelu oraz detekcję zjawiska przeuczenia.
        """
        # Skalowanie osi czasu zgodnie z rzeczywistym numerowaniem epok (od 1)
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(15, 6))

        # Sekcja wizualizacji funkcji straty (Loss)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, marker='o', label='Training Loss')
        plt.plot(epochs, val_losses, marker='s', label='Validation Loss')
        plt.title(f'Fold {fold}: Learning Curves (Loss)', fontsize=14, pad=15)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.xticks(epochs)

        # Sekcja wizualizacji dokładności (Accuracy)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, marker='o', label='Training Accuracy')
        plt.plot(epochs, val_accuracies, marker='s', label='Validation Accuracy')
        plt.title(f'Fold {fold}: Learning Curves (Accuracy)', fontsize=14, pad=15)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.xticks(epochs)

        plt.tight_layout()

        # Zapis w wysokiej gęstości punktów (300 DPI) dla zachowania jakości w publikacjach
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Renderowanie w trybie nieblokującym, umożliwiające kontynuację obliczeń w tle
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(cm, class_names, fold, save_path):
        """
        Generuje znormalizowaną macierz pomyłek w formie mapy ciepła.
        Umożliwia szczegółową diagnostykę błędów klasyfikacji dla poszczególnych etykiet.
        """
        # Normalizacja wierszowa macierzy do wartości procentowych
        cm_percent = cm.float() / cm.sum(dim=1, keepdim=True)
        cm_percent_np = cm_percent.numpy() * 100

        plt.figure(figsize=(16, 14))

        # Generowanie mapy ciepła z adnotacjami numerycznymi (Precision/Recall visualization)
        sns.heatmap(
            cm_percent_np,
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="viridis",
            annot=True,  # Wyświetlanie wartości procentowych na polach macierzy
            fmt=".1f",  # Ograniczenie precyzji do jednego miejsca po przecinku
            cbar_kws={'label': 'Accuracy (%)'}
        )

        plt.title(f"Fold {fold}: Confusion Matrix (Normalized %)", fontsize=16, pad=20)

        # Optymalizacja czytelności dla dużych zbiorów klas (rotacja etykiet 45°)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()

        # Eksport grafiki w standardzie High-Resolution
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(1)
        plt.close()