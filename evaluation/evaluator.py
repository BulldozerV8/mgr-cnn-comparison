import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from evaluation.visualization import Visualizer


class ModelEvaluator:
    """
    Klasa odpowiedzialna za kompleksową ewaluację statystyczną i diagnostyczną modelu.
    Realizuje proces obliczania metryk, analizę błędów oraz selekcję próbek do badań
    interpretowalności (Grad-CAM), dbając o minimalne zużycie zasobów sprzętowych.
    """

    def __init__(self, num_classes, device, config):
        """
        Inicjalizacja modułu ewaluacji z optymalizacją alokacji zasobów.
        Metryki są tworzone raz i przechowywane w pamięci urządzenia (GPU/CPU),
        co eliminuje narzut czasowy podczas wielokrotnych wywołań w pętli CV.
        """
        self.num_classes = num_classes
        self.device = device
        self.config = config

        # Centralizacja metryk w ModuleDict dla łatwego zarządzania i transferu na urządzenie
        self.metrics = torch.nn.ModuleDict({
            "acc": Accuracy(task="multiclass", num_classes=num_classes),
            "prec": Precision(task="multiclass", num_classes=num_classes,
                              average=config.get("metrics_average", "macro")),
            "rec": Recall(task="multiclass", num_classes=num_classes, average=config.get("metrics_average", "macro")),
            "f1": F1Score(task="multiclass", num_classes=num_classes, average=config.get("metrics_average", "macro")),
            "cm": ConfusionMatrix(task="multiclass", num_classes=num_classes)
        }).to(device)

    def evaluate(self, model, test_loader, criterion, class_names, fold, cm_save_path):
        """
        Główna procedura testowa. Przeprowadza inferencję na zbiorze walidacyjnym/testowym,
        agreguje wyniki i zarządza procesem generowania raportów wizualnych.
        """
        # Przygotowanie metryk do nowego cyklu obliczeniowego
        for metric in self.metrics.values():
            metric.reset()

        # Konfiguracja buforów diagnostycznych (Memory-Efficient Tracking)
        top_n = self.config.get("num_visualizations", 5)
        top_successes = []
        top_failures = []

        total_loss, num_batches = 0, 0
        model.eval()

        with torch.no_grad():
            for batch_idx, (images, labels, paths) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1

                # Ekstrakcja pewności predykcji (Confidence Score) do analizy diagnostycznej
                probs = F.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probs, 1)

                # Aktualizacja buforów próbek ekstremalnych bez obciążania pamięci RAM
                self._update_buffers(top_successes, top_failures, confidences, predicted, labels, paths, top_n)

                # Masowa aktualizacja stanów metryk
                for metric in self.metrics.values():
                    metric.update(predicted, labels)

        # Finalna agregacja wyników (obliczenie wartości końcowych z buforów metryk)
        computed = {k: v.compute() for k, v in self.metrics.items()}
        avg_loss = total_loss / num_batches

        # Przekazanie danych do logowania i generowania wykresów
        return self._process_results(computed, avg_loss, class_names, fold, cm_save_path, top_successes, top_failures)

    def _update_buffers(self, successes, failures, confs, preds, targets, paths, top_n):
        """
        Implementacja selektywnego buforowania próbek. Zamiast przechowywać dane o wszystkich
        zdjęciach, metoda utrzymuje jedynie stałą liczbę próbek o najwyższej pewności predykcji,
        co gwarantuje stały koszt pamięciowy niezależnie od wielkości zbioru danych.
        """
        for i in range(len(targets)):
            conf = confs[i].item()
            is_correct = (preds[i] == targets[i]).item()
            sample = {
                'confidence': conf,
                'pred_idx': preds[i].item(),
                'true_idx': targets[i].item(),
                'correct': is_correct,
                'path': paths[i]
            }

            # Wybór odpowiedniej listy (sukcesy vs błędy)
            target_list = successes if is_correct else failures

            # Logika utrzymania TOP-N najbardziej pewnych przypadków
            if len(target_list) < top_n:
                target_list.append(sample)
                target_list.sort(key=lambda x: x['confidence'], reverse=True)
            elif conf > target_list[-1]['confidence']:
                target_list[-1] = sample
                target_list.sort(key=lambda x: x['confidence'], reverse=True)

    def _process_results(self, computed, loss, class_names, fold, cm_path, successes, failures):
        """
        Moduł przetwarzania końcowego. Odpowiada za logowanie statystyk do konsoli,
        delegowanie generowania macierzy pomyłek oraz archiwizację metryk kluczowych.
        """
        acc = computed['acc'].item()
        prec = computed['prec'].item()
        rec = computed['rec'].item()
        f1 = computed['f1'].item()
        cm = computed['cm'].cpu()

        # Raportowanie wydajności w czasie rzeczywistym
        print(f"\nFold {fold} Test Loss: {loss:.6f}")
        print(f"Fold {fold} Test Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

        # Delegacja wizualizacji macierzy pomyłek (Separation of Concerns)
        Visualizer.plot_confusion_matrix(cm, class_names, fold, cm_path)

        # Szczegółowa analiza błędów klasyfikacji
        errors = []
        for t in range(self.num_classes):
            for p in range(self.num_classes):
                if t != p and cm[t, p] > 0:
                    errors.append((int(cm[t, p]), t, p))

        errors.sort(reverse=True, key=lambda x: x[0])
        print(f"\nTOP 10 najczęstszych pomyłek dla Fold {fold}:\n")
        for i in range(min(10, len(errors))):
            count, t, p = errors[i]
            print(f"{i + 1}. {class_names[t]} → {class_names[p]} : {count} razy")

        # Statystyczne podsumowanie pewności modelu
        print(f"\nAnaliza pewności decyzji (Fold {fold}):")
        if successes:
            print(
                f"  - Najpewniejszy sukces: {successes[0]['confidence'] * 100:.2f}% ({class_names[successes[0]['true_idx']]})")
        if failures:
            print(
                f"  - Najbardziej pewny błąd: {failures[0]['confidence'] * 100:.2f}% (Model myślał: {class_names[failures[0]['pred_idx']]}, było: {class_names[failures[0]['true_idx']]})")

        print(f"\nFold {fold} Test Acc: {acc:.4f} | F1: {f1:.4f}")

        return {
            'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
            'top_successes': successes, 'top_failures': failures
        }