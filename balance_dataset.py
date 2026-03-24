import os
import shutil
import random

# Ustawiamy ziarno losowości, żeby za każdym razem losowało tak samo
random.seed(42)

source_folder = "Dataset-small/"  # Ścieżka do Twojego obecnego folderu
dest_folder = "Dataset-small-perfect/"  # Tu powstanie zbalansowana baza
MAX_IMAGES = 400

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

classes = os.listdir(source_folder)

for cls in classes:
    cls_path = os.path.join(source_folder, cls)
    if not os.path.isdir(cls_path): continue

    dest_cls_path = os.path.join(dest_folder, cls)
    os.makedirs(dest_cls_path, exist_ok=True)

    images = os.listdir(cls_path)
    random.shuffle(images)

    # Bierzemy max 400 zdjęć (lub wszystkie, jeśli jest ich mniej)
    selected_images = images[:MAX_IMAGES]

    print(f"Klasa {cls}: kopiowanie {len(selected_images)} zdjęć...")
    for img in selected_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(dest_cls_path, img)
        shutil.copy(src, dst)

print("\nGotowe! Nowa baza znajduje się w folderze 'Dataset-perfect/'.")