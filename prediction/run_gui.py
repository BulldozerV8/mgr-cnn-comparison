import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

# --- KONIECZNA ZMIANA DLA IMPORTÓW ---
# Dodajemy główny katalog projektu do ścieżki wyszukiwania modułów
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prediction.predictor import Predictor
from config import CONFIG


class PredictionGUI:
    """
    Profesjonalny interfejs graficzny (GUI) dla silnika predykcyjnego.
    Umożliwia interaktywną analizę obrazów, wizualizację pewności modelu
    oraz monitorowanie wydajności inferencji w czasie rzeczywistym.
    """

    def __init__(self, root):
        self.root = root
        self.root.title(f"AI Prediction Engine - {CONFIG['test_name']}")
        self.root.geometry("900x650")
        self.root.configure(bg="#f0f0f0")

        # --- 1. Inicjalizacja Silnika ---
        # Ścieżka do modelu: wychodzimy poziom wyżej (..), bo skrypt jest w podfolderze
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        #test_folder = os.path.join(project_root, "results", CONFIG["test_name"])
        #model_path = os.path.join(project_root, "results/1_Final_testy_ptaki/FINAL_Ptaki_448px_CV5/saved_models/model_weights_fold_5.pth")
        #model_path = os.path.join(project_root, "results/1_Final_testy_samoloty/FINAL_Samoloty_448px_CV5/saved_models/model_weights_fold_4.pth")
        model_path = os.path.join(project_root, "results/Custom_Final_Ptaki_448px/saved_models/model_weights_fold_2.pth")
        #model_path = os.path.join(project_root, "results/Custom_Final_Samoloty_448px/saved_models/model_weights_fold_5.pth")
        #model_path = os.path.join(test_folder, "saved_models", "model_weights_fold_1.pth")

        try:
            self.predictor = Predictor(model_path, device=CONFIG["device"])
        except Exception as e:
            messagebox.showerror("Błąd inicjalizacji", f"Nie można załadować modelu pod ścieżką:\n{model_path}\n\nBłąd: {e}")
            self.root.destroy()
            return

        self._setup_ui()

    def _setup_ui(self):
        """Tworzy strukturę wizualną aplikacji."""
        # Główny kontener
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Lewy panel: Podgląd obrazu
        self.left_panel = tk.LabelFrame(main_frame, text=" Podgląd Obrazu ", bg="#f0f0f0", font=("Arial", 10, "bold"))
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        self.img_label = tk.Label(self.left_panel, text="Wybierz zdjęcie, aby rozpocząć", bg="#d9d9d9")
        self.img_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Prawy panel: Wyniki i Kontrola
        right_panel = tk.Frame(main_frame, bg="#f0f0f0")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Przycisk ładowania
        btn_load = tk.Button(right_panel, text="WCZYTAJ ZDJĘCIE", command=self.load_and_predict,
                             bg="#2ecc71", fg="white", font=("Arial", 11, "bold"), height=2, width=20)
        btn_load.pack(pady=10)

        # Sekcja wyników Top-3
        self.results_frame = tk.LabelFrame(right_panel, text=" Wyniki Analizy (Top-3) ", bg="#f0f0f0",
                                           font=("Arial", 10, "bold"))
        self.results_frame.pack(fill=tk.X, pady=10)

        self.prob_bars = []
        self.prob_labels = []
        for i in range(3):
            label = tk.Label(self.results_frame, text=f"{i + 1}. ---", bg="#f0f0f0", anchor="w")
            label.pack(fill=tk.X, padx=10, pady=(5, 0))

            bar = ttk.Progressbar(self.results_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
            bar.pack(fill=tk.X, padx=10, pady=(0, 10))

            self.prob_labels.append(label)
            self.prob_bars.append(bar)

        # Metryki wydajności
        self.metrics_label = tk.Label(right_panel, text="Czas inferencji: --- ms", bg="#f0f0f0", font=("Arial", 9))
        self.metrics_label.pack(pady=5)

        # Dolny panel: Konsola logów
        log_frame = tk.LabelFrame(self.root, text=" Logi Systemowe ", bg="#f0f0f0", font=("Arial", 9, "bold"))
        log_frame.pack(fill=tk.X, padx=30, pady=(0, 20))

        self.log_text = tk.Text(log_frame, height=5, state=tk.DISABLED, bg="black", fg="#00ff00", font=("Consolas", 9))
        self.log_text.pack(fill=tk.X, padx=5, pady=5)

    def log(self, message):
        """Dodaje wpis do konsoli wewnątrz GUI."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"> {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def load_and_predict(self):
        """Obsługuje wybór pliku i proces predykcji."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        self.log(f"Analiza: {os.path.basename(file_path)}")

        # 1. Wyświetlenie obrazu
        img = Image.open(file_path)
        img.thumbnail((450, 450))  # Skalowanie do podglądu
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk, text="")
        self.img_label.image = img_tk

        # 2. Predykcja
        result = self.predictor.predict(file_path, top_k=3)

        if result["status"] == "success":
            # Aktualizacja UI wynikami
            self.metrics_label.config(text=f"Czas inferencji: {result['time_ms']:.2f} ms")

            for i, res in enumerate(result["top_k"]):
                conf_pct = res['confidence'] * 100
                self.prob_labels[i].config(text=f"{i + 1}. {res['label']} ({conf_pct:.2f}%)")
                self.prob_bars[i]['value'] = conf_pct

            self.log(f"Rozpoznano: {result['top_k'][0]['label']} ({result['top_k'][0]['confidence'] * 100:.1f}%)")
        else:
            self.log(f"BŁĄD: {result.get('message')}")
            messagebox.showerror("Błąd", result.get("message"))


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionGUI(root)
    root.mainloop()