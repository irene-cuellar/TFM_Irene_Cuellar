#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:02:37 2026

@author: irene
"""

"""
ADNI image resize script.

This script prepares ADNI PNG images for model input by resizing them to
224x224 pixels. It processes PVE registered images, coronal slice images,
and axial slice images. PVE and coronal images are resized directly, while
axial images are resized preserving the aspect ratio and then center cropped.
"""

from pathlib import Path
from PIL import Image


# =========================================================
# RUTAS
# =========================================================

# PVE registered images
PVE_INPUT_DIR = Path("/pool/home/AD_Multimodal/ADNI/irene_adni/pve_registered")
PVE_OUTPUT_DIR = Path("/pool/home/AD_Multimodal/ADNI/irene_adni/pve_registered/resized")

# Coronal slices images
CORONAL_INPUT_ROOT = Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal_slices")
CORONAL_OUTPUT_ROOT = Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal_slices_resized")

# Axial images
AXIAL_CONFIGS = [
    {
        "name": "axial central",
        "input_dir": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_central"),
        "output_crop_dir": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_central_crop"),
    },
    {
        "name": "axial minus1",
        "input_dir": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_minus1"),
        "output_crop_dir": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_minus1_crop"),
    },
    {
        "name": "axial plus1",
        "input_dir": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_plus1"),
        "output_crop_dir": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_plus1_crop"),
    },
]


# =========================================================
# PARÁMETROS
# =========================================================

TARGET_SIZE = (224, 224)

EXPECTED_SQUARE_INPUT_SIZE = (182, 182)  # PVE and coronal images
EXPECTED_AXIAL_INPUT_SIZE = (182, 218)   # Axial images

VALID_EXT = ".png"

EXPECTED_SUBJECTS = 902
EXPECTED_PNGS_PER_SUBJECT = 20

ADD_RESIZED_SUFFIX = True


# =========================================================
# FUNCIONES GENERALES
# =========================================================

def resize_direct(input_path, output_path, expected_size, target_size):
    """
    Opens a PNG image, converts it to grayscale, resizes it directly,
    and saves it.

    Returns:
        different_size: True if the original size is different from expected_size.
        original_size: original image size.
    """

    with Image.open(input_path) as img:
        original_size = img.size
        different_size = original_size != expected_size

        img = img.convert("L")
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        img_resized.save(output_path)

    return different_size, original_size


def resize_with_center_crop(img, target_size=(224, 224)):
    """
    Resize preserving the aspect ratio so that the image fully covers 224x224,
    then applies a center crop.
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
# PROCESADO DE PVE REGISTERED
# =========================================================

def process_pve_images():
    print("\n" + "=" * 70)
    print("Procesando imágenes PVE registered")
    print("=" * 70)

    if not PVE_INPUT_DIR.exists():
        print(f"AVISO: no existe la carpeta de entrada: {PVE_INPUT_DIR}")
        return

    PVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted([
        p for p in PVE_INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == VALID_EXT
    ])

    print(f"Se han encontrado {len(files)} imágenes en: {PVE_INPUT_DIR}")

    ok_count = 0
    error_count = 0
    different_size_count = 0

    for i, input_path in enumerate(files, start=1):
        try:
            if ADD_RESIZED_SUFFIX:
                output_filename = f"{input_path.stem}_resized{input_path.suffix}"
            else:
                output_filename = input_path.name

            output_path = PVE_OUTPUT_DIR / output_filename

            different_size, original_size = resize_direct(
                input_path=input_path,
                output_path=output_path,
                expected_size=EXPECTED_SQUARE_INPUT_SIZE,
                target_size=TARGET_SIZE,
            )

            if different_size:
                different_size_count += 1
                print(
                    f"AVISO [{i}/{len(files)}] {input_path.name}: "
                    f"tamaño original {original_size}, esperado {EXPECTED_SQUARE_INPUT_SIZE}."
                )

            ok_count += 1
            print(f"OK [{i}/{len(files)}]: {input_path.name}")

        except Exception as e:
            error_count += 1
            print(f"ERROR [{i}/{len(files)}] con {input_path.name}: {e}")

    print("\nResumen PVE registered:")
    print(f"Imágenes procesadas correctamente: {ok_count}")
    print(f"Imágenes con error: {error_count}")
    print(f"Imágenes con tamaño distinto a {EXPECTED_SQUARE_INPUT_SIZE}: {different_size_count}")
    print(f"Salida resized: {PVE_OUTPUT_DIR}")


# =========================================================
# PROCESADO DE CORONAL SLICES
# =========================================================

def process_coronal_slices():
    print("\n" + "=" * 70)
    print("Procesando imágenes coronal slices")
    print("=" * 70)

    if not CORONAL_INPUT_ROOT.exists():
        print(f"AVISO: no existe la carpeta de entrada: {CORONAL_INPUT_ROOT}")
        return

    CORONAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([
        p for p in CORONAL_INPUT_ROOT.iterdir()
        if p.is_dir()
    ])

    print(f"Se han encontrado {len(subject_dirs)} carpetas de sujetos en: {CORONAL_INPUT_ROOT}")

    if len(subject_dirs) != EXPECTED_SUBJECTS:
        print(
            f"AVISO: se esperaban {EXPECTED_SUBJECTS} carpetas de sujetos, "
            f"pero se han encontrado {len(subject_dirs)}."
        )

    images_by_subject = {}
    total_images = 0

    for subject_dir in subject_dirs:
        png_files = sorted([
            p for p in subject_dir.iterdir()
            if p.is_file() and p.suffix.lower() == VALID_EXT
        ])

        images_by_subject[subject_dir] = png_files
        total_images += len(png_files)

    print(f"Se han encontrado {total_images} imágenes PNG en total.")

    ok_count = 0
    error_count = 0
    different_size_count = 0
    subject_warning_count = 0
    image_index = 0

    for subject_index, subject_dir in enumerate(subject_dirs, start=1):
        subject_id = subject_dir.name
        output_subject_dir = CORONAL_OUTPUT_ROOT / subject_id
        output_subject_dir.mkdir(parents=True, exist_ok=True)

        png_files = images_by_subject[subject_dir]

        if len(png_files) != EXPECTED_PNGS_PER_SUBJECT:
            subject_warning_count += 1
            print(
                f"AVISO sujeto [{subject_index}/{len(subject_dirs)}] {subject_id}: "
                f"tiene {len(png_files)} PNG, esperado {EXPECTED_PNGS_PER_SUBJECT}."
            )

        for input_path in png_files:
            image_index += 1

            try:
                if ADD_RESIZED_SUFFIX:
                    output_filename = f"{input_path.stem}_resized{input_path.suffix}"
                else:
                    output_filename = input_path.name

                output_path = output_subject_dir / output_filename

                different_size, original_size = resize_direct(
                    input_path=input_path,
                    output_path=output_path,
                    expected_size=EXPECTED_SQUARE_INPUT_SIZE,
                    target_size=TARGET_SIZE,
                )

                if different_size:
                    different_size_count += 1
                    print(
                        f"AVISO imagen [{image_index}/{total_images}] {input_path}: "
                        f"tamaño original {original_size}, esperado {EXPECTED_SQUARE_INPUT_SIZE}."
                    )

                ok_count += 1
                print(f"OK [{image_index}/{total_images}]: {subject_id}/{input_path.name}")

            except Exception as e:
                error_count += 1
                print(f"ERROR [{image_index}/{total_images}] con {input_path}: {e}")

    print("\nResumen coronal slices:")
    print(f"Carpetas de sujetos encontradas: {len(subject_dirs)}")
    print(f"Sujetos con número de PNG distinto a {EXPECTED_PNGS_PER_SUBJECT}: {subject_warning_count}")
    print(f"Imágenes PNG encontradas: {total_images}")
    print(f"Imágenes procesadas correctamente: {ok_count}")
    print(f"Imágenes con error: {error_count}")
    print(f"Imágenes con tamaño distinto a {EXPECTED_SQUARE_INPUT_SIZE}: {different_size_count}")
    print(f"Salida resized: {CORONAL_OUTPUT_ROOT}")


# =========================================================
# PROCESADO DE AXIAL IMAGES
# =========================================================

def process_axial_images():
    print("\n" + "=" * 70)
    print("Procesando imágenes axial")
    print("=" * 70)

    total_ok = 0
    total_errors = 0
    total_different_size = 0

    for config in AXIAL_CONFIGS:
        name = config["name"]
        input_dir = config["input_dir"]
        output_crop_dir = config["output_crop_dir"]

        print("\n" + "-" * 70)
        print(f"Procesando: {name}")
        print("-" * 70)

        if not input_dir.exists():
            print(f"AVISO: no existe la carpeta de entrada: {input_dir}")
            continue

        output_crop_dir.mkdir(parents=True, exist_ok=True)

        files = sorted([
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() == VALID_EXT
        ])

        print(f"Se han encontrado {len(files)} imágenes en: {input_dir}")

        ok_count = 0
        error_count = 0
        different_size_count = 0

        for i, input_path in enumerate(files, start=1):
            try:
                with Image.open(input_path) as img:
                    original_size = img.size

                    if original_size != EXPECTED_AXIAL_INPUT_SIZE:
                        different_size_count += 1
                        print(
                            f"AVISO [{i}/{len(files)}] {input_path.name}: "
                            f"tamaño original {original_size}, esperado {EXPECTED_AXIAL_INPUT_SIZE}."
                        )

                    img = img.convert("L")

                    img_crop = resize_with_center_crop(
                        img,
                        target_size=TARGET_SIZE,
                    )

                    output_crop_path = output_crop_dir / input_path.name
                    img_crop.save(output_crop_path)

                ok_count += 1
                print(f"OK [{i}/{len(files)}]: {input_path.name}")

            except Exception as e:
                error_count += 1
                print(f"ERROR [{i}/{len(files)}] con {input_path.name}: {e}")

        total_ok += ok_count
        total_errors += error_count
        total_different_size += different_size_count

        print(f"\nResumen {name}:")
        print(f"Imágenes procesadas correctamente: {ok_count}")
        print(f"Imágenes con error: {error_count}")
        print(f"Imágenes con tamaño distinto a {EXPECTED_AXIAL_INPUT_SIZE}: {different_size_count}")
        print(f"Salida crop: {output_crop_dir}")

    print("\nResumen axial total:")
    print(f"Imágenes procesadas correctamente: {total_ok}")
    print(f"Imágenes con error: {total_errors}")
    print(f"Imágenes con tamaño distinto a {EXPECTED_AXIAL_INPUT_SIZE}: {total_different_size}")


# =========================================================
# EJECUCIÓN PRINCIPAL
# =========================================================

def main():
    process_pve_images()
    process_coronal_slices()
    process_axial_images()

    print("\n" + "=" * 70)
    print("Proceso completo terminado.")
    print("=" * 70)


if __name__ == "__main__":
    main()