import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os


class GradCAM:
    """
    Zoptymalizowana klasa do generowania map ciepła Grad-CAM.
    Obsługuje automatyczne czyszczenie hooków i optymalizację pamięci.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Rejestruje i przechowuje referencje do hooków, aby móc je później usunąć."""
        self.hooks.append(self.target_layer.register_forward_hook(self._save_activations))
        self.hooks.append(self.target_layer.register_full_backward_hook(self._save_gradients))

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Usuwa hooki z modelu. Kluczowe dla uniknięcia wycieków pamięci w CV."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_heatmap(self, input_tensor, class_idx=None):
        """Generuje mapę ciepła dla zadanej klasy."""
        self.model.eval()  # Gwarancja, że BatchNorm/Dropout nie wpływają na wynik

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward pass
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # GAP (Global Average Pooling) na gradientach
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Liniowa kombinacja aktywacji i wag (ważona suma aktywacji)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        # ReLU - interesują nas tylko cechy zwiększające prawdopodobieństwo danej klasy
        cam = F.relu(cam)

        # Normalizacja z zabezpieczeniem przed dzieleniem przez zero
        cam = cam.detach().cpu().numpy()
        denom = (cam.max() - cam.min()) + 1e-10
        cam = (cam - cam.min()) / denom

        return cam

    def __del__(self):
        """Destruktor zapewniający usunięcie hooków przy usuwaniu obiektu."""
        self.remove_hooks()


class CAMVisualizer:
    """Szybki wizualizator wykorzystujący natywne funkcje OpenCV."""

    @staticmethod
    def save_visualization(img_path, heatmap, save_path, alpha=0.6):
        # Wczytanie obrazu
        img = cv2.imread(img_path)
        if img is None:
            return

        # Przeskalowanie heatmapy do rozmiaru oryginału
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Konwersja na format 8-bitowy i nałożenie mapy kolorów JET
        heatmap_8bit = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)

        # Połączenie obrazów (Superimposition)
        # alpha * img + (1-alpha) * heatmap
        superimposed_img = cv2.addWeighted(img, alpha, heatmap_color, 1 - alpha, 0)

        # Bezpośredni zapis przez OpenCV (dużo szybszy niż Matplotlib)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, superimposed_img)