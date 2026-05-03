class EarlyStopping:
    """
    Mechanizm wczesnego zatrzymywania treningu (Early Stopping).
    Monitoruje stratę walidacyjną i przerywa proces, gdy model przestaje robić postępy,
    zapobiegając tym samym zjawisku przeuczenia (overfitting).
    """

    def __init__(self, patience=7, min_delta=0):
        """
        Inicjalizacja parametrów monitorowania.

        Args:
            patience (int): Liczba epok czekania na poprawę przed zatrzymaniem.
            min_delta (float): Minimalna zmiana straty uznawana za realną poprawę.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.early_stop = False
        self.improved = False  # Flaga informująca, czy w tej epoce mamy rekord

    def __call__(self, val_loss, epoch):
        """
        Główna logika sprawdzająca postęp w każdej epoce.
        """
        self.improved = False  # Reset flagi przy każdym wywołaniu

        # Przypadek dla pierwszej epoki (inicjalizacja rekordów)
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.improved = True  # Pierwszy model jest zawsze najlepszy

        # Jeśli strata nie spadła wystarczająco (brak poprawy)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"    [EarlyStopping] Brak poprawy. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True  # Sygnał do przerwania pętli głównej

        # Jeśli nastąpiła realna poprawa wyników
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.improved = True  # Sygnał dla skryptu głównego: "ZAPISZ WAGI!"
            print(f"    [EarlyStopping] Nowy najlepszy Loss: {val_loss:.4f} (Epoka: {epoch})")

        return self.early_stop