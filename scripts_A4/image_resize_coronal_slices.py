#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:36:30 2026

@author: irene
"""
"""
Resize de las 20 slices centrales de la vista coronal del estudio A4.

Estructura de entrada:
coronal/
├── sujeto_001/
│   ├── slice_01.png
│   └── ...
├── sujeto_002/
│   ├── slice_01.png
│   └── ...

Estructura de salida:
coronal_resized/
├── sujeto_001/
│   ├── slice_01.png
│   └── ...
├── sujeto_002/
│   ├── slice_01.png
│   └── ...

input shape  --> 197x189
output shape --> 224x224
técnica: resize + center crop
"""

import os
from PIL import Image

# =========================================================
# RUTAS
# =========================================================
input_dir = "/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/coronal"

output_dir = "/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/coronal_resized"

os.makedirs(output_dir, exist_ok=True)

target_size = (224, 224)

# =========================================================
# FUNCIÓN
# =========================================================
def resize_with_center_crop(img, target_size=(224, 224)):
    """
    Resize manteniendo aspecto para cubrir completamente 224x224
    y luego hace center crop.
    """
    target_w, target_h = target_size
    w, h = img.size

    scale = max(target_w / w, target_h / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    img_cropped = img_resized.crop((left, top, right, bottom))

    return img_cropped


# =========================================================
# PROCESADO
# =========================================================
subject_folders = sorted(os.listdir(input_dir))

print(f"Se han encontrado {len(subject_folders)} sujetos en: {input_dir}")

total_images = 0
ok_count = 0
error_count = 0

for subject in subject_folders:
    subject_input_dir = os.path.join(input_dir, subject)
    subject_output_dir = os.path.join(output_dir, subject)

    os.makedirs(subject_output_dir, exist_ok=True)

    png_files = sorted([
        f for f in os.listdir(subject_input_dir)
        if f.lower().endswith(".png")
    ])

    total_images += len(png_files)

    for filename in png_files:
        input_path = os.path.join(subject_input_dir, filename)
        output_path = os.path.join(subject_output_dir, filename)

        try:
            with Image.open(input_path) as img:
                img = img.convert("L")  # grayscale

                img_resized = resize_with_center_crop(
                    img,
                    target_size=target_size
                )

                img_resized.save(output_path)

            ok_count += 1

        except Exception as e:
            error_count += 1
            print(f"ERROR con {subject}/{filename}: {e}")

    print(f"OK sujeto {subject}: {len(png_files)} imágenes procesadas")


print("\nProceso terminado.")
print(f"Sujetos procesados: {len(subject_folders)}")
print(f"Imágenes encontradas: {total_images}")
print(f"Imágenes procesadas correctamente: {ok_count}")
print(f"Imágenes con error: {error_count}")
print(f"Salida: {output_dir}")