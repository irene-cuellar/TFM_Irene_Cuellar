#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:09:09 2026

@author: irene
"""
"""
A4 - Coronal slices resized (las 20 centrales)

Normalización gaussiana para todas las imágenes PNG dentro de:
   /pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/coronal_resized

La carpeta de entrada contiene subcarpetas, una por sujeto.
El script mantiene la misma estructura de carpetas en las salidas.

Entrada:
    coronal_resized/
        sujeto_1/
            imagen_1.png
            ...
        sujeto_2/
            imagen_1.png
            ...

Salidas:
    coronal_resized_smooth/
        sujeto_1/
            imagen_1_smooth.png
            ...

    coronal_resized_final/
        sujeto_1/
            imagen_1_final.png
            ...
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


# =========================================================
# RUTAS
# =========================================================

input_root = Path("/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/coronal_resized")

smooth_output_root = Path(
    "/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/coronal_resized_smooth"
)

final_output_root = Path(
    "/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/coronal_resized_final"
)


# =========================================================
# PARÁMETROS
# =========================================================

# FWHM en mm.
# Si la resolución es 1 x 1 x 1 mm, equivale a píxeles en 2D.
FWHM_MM = 6.0

# Conversión FWHM a sigma:
# FWHM = 2.355 * sigma
SIGMA = FWHM_MM / 2.355

# Píxeles por encima de este valor se consideran tejido útil.
BACKGROUND_THRESHOLD = 0.0

valid_ext = ".png"

expected_subjects = 1240
expected_pngs_per_subject = 20


# =========================================================
# FUNCIONES
# =========================================================

def load_png_as_float(path):
    """
    Carga una imagen PNG y la convierte a float32 en rango [0, 1].
    """

    img = Image.open(path)

    # Convertir a escala de grises si hace falta.
    if img.mode not in ["L", "I;16", "I", "F"]:
        img = img.convert("L")

    arr = np.array(img)

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0

    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0

    else:
        arr = arr.astype(np.float32)

        if arr.max() > 1.0:
            arr = arr / arr.max()

    return arr


def save_float_png(path, arr):
    """
    Guarda una imagen float en rango [0, 1] como PNG de 8 bits.
    """

    arr = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr * 255).round().astype(np.uint8)

    img = Image.fromarray(arr_uint8)
    img.save(path)


def process_image(image_path, smooth_output_dir, final_output_dir):
    """
    Procesa una imagen:
    1. Carga la imagen original.
    2. Aplica suavizado gaussiano.
    3. Calcula vmin y vmax sobre la imagen suavizada dentro de la máscara.
    4. Normaliza la imagen original usando esos vmin y vmax.
    5. Guarda imagen suavizada e imagen final.
    """

    image_path = Path(image_path)
    smooth_output_dir = Path(smooth_output_dir)
    final_output_dir = Path(final_output_dir)

    original = load_png_as_float(image_path)

    # Máscara de tejido útil usando la imagen original.
    mask = original > BACKGROUND_THRESHOLD

    if not np.any(mask):
        return False, "no useful pixels found"

    # Suavizado gaussiano.
    smooth = gaussian_filter(original, sigma=SIGMA)

    # vmin y vmax de la imagen suavizada, solo dentro de la máscara.
    vals = smooth[mask]
    vmin = vals.min()
    vmax = vals.max()

    if vmax <= vmin:
        return False, "vmax <= vmin"

    # Normalizar imagen original usando vmin y vmax de la imagen suavizada.
    final = (original - vmin) / (vmax - vmin)
    final = np.clip(final, 0.0, 1.0)

    stem = image_path.stem

    smooth_path = smooth_output_dir / f"{stem}_smooth.png"
    final_path = final_output_dir / f"{stem}_final.png"

    save_float_png(smooth_path, smooth)
    save_float_png(final_path, final)

    return True, None


def process_subject_folder(subject_dir):
    """
    Procesa todos los PNG de una carpeta de sujeto.
    Mantiene la misma carpeta de sujeto en las salidas.
    """

    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name

    smooth_subject_dir = smooth_output_root / subject_id
    final_subject_dir = final_output_root / subject_id

    smooth_subject_dir.mkdir(parents=True, exist_ok=True)
    final_subject_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted([
        p for p in subject_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == valid_ext
        and not p.name.endswith("_smooth.png")
        and not p.name.endswith("_final.png")
    ])

    if len(png_files) != expected_pngs_per_subject:
        print(
            f"AVISO sujeto {subject_id}: "
            f"tiene {len(png_files)} PNG, esperado {expected_pngs_per_subject}."
        )

    skipped = []
    ok_count = 0

    if tqdm is not None:
        iterator = tqdm(png_files, desc=subject_id, unit="img")
    else:
        iterator = png_files

    for i, image_path in enumerate(iterator, start=1):
        success, reason = process_image(
            image_path=image_path,
            smooth_output_dir=smooth_subject_dir,
            final_output_dir=final_subject_dir,
        )

        if success:
            ok_count += 1

            if tqdm is None:
                print(f"OK {subject_id} [{i}/{len(png_files)}]: {image_path.name}")
        else:
            skipped.append((image_path.name, reason))

            if tqdm is None:
                print(
                    f"SKIPPED {subject_id} [{i}/{len(png_files)}]: "
                    f"{image_path.name} - {reason}"
                )

    return ok_count, skipped, len(png_files)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print(f"Using FWHM = {FWHM_MM} mm")
    print(f"Using sigma = {SIGMA:.4f} pixels")

    if tqdm is None:
        print("AVISO: tqdm no está instalado. Se usará progreso con print.")

    if not input_root.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {input_root}")

    smooth_output_root.mkdir(parents=True, exist_ok=True)
    final_output_root.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([
        p for p in input_root.iterdir()
        if p.is_dir()
    ])

    print(f"\nCarpeta de entrada: {input_root}")
    print(f"Carpeta smooth:     {smooth_output_root}")
    print(f"Carpeta final:      {final_output_root}")
    print(f"Sujetos encontrados: {len(subject_dirs)}")

    if len(subject_dirs) != expected_subjects:
        print(
            f"AVISO: se esperaban {expected_subjects} sujetos, "
            f"pero se han encontrado {len(subject_dirs)}."
        )

    total_ok = 0
    total_images = 0
    total_skipped = []
    subject_warning_count = 0

    for subject_index, subject_dir in enumerate(subject_dirs, start=1):
        print("\n======================================")
        print(f"Sujeto [{subject_index}/{len(subject_dirs)}]: {subject_dir.name}")
        print("======================================")

        ok_count, skipped, n_images = process_subject_folder(subject_dir)

        total_ok += ok_count
        total_images += n_images

        if n_images != expected_pngs_per_subject:
            subject_warning_count += 1

        for image_name, reason in skipped:
            total_skipped.append((subject_dir.name, image_name, reason))

    print("\nPipeline finished.")
    print(f"Sujetos procesados: {len(subject_dirs)}")
    print(f"Sujetos con número de PNG distinto a {expected_pngs_per_subject}: {subject_warning_count}")
    print(f"Imágenes encontradas: {total_images}")
    print(f"Imágenes procesadas correctamente: {total_ok}")
    print(f"Imágenes saltadas: {len(total_skipped)}")

    if total_skipped:
        print("\nImágenes saltadas:")
        for subject_id, image_name, reason in total_skipped:
            print(f"  {subject_id}/{image_name}: {reason}")

    print(f"\nSalida smooth: {smooth_output_root}")
    print(f"Salida final:  {final_output_root}")