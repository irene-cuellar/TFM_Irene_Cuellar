#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: irene
"""

"""
ADNI Gaussian normalization script.

This script applies Gaussian-based intensity normalization to ADNI PNG images.
For each image, it creates a smoothed version and a final normalized version.
The normalization uses the minimum and maximum values from the smoothed image,
computed only over non-background pixels, and applies this scaling to the
original image.

The script processes:
    - axial crop images
    - coronal patch images
    - resized coronal slices stored in subject-level folders
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# tqdm is optional. If it is not installed, the script will still work.
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


# =========================================================
# PARAMETERS
# =========================================================

# FWHM in mm.
# If the image resolution is 1 x 1 x 1 mm, this is equivalent to pixels in 2D.
FWHM_MM = 6.0

# Convert FWHM to sigma:
# FWHM = 2.355 * sigma
SIGMA = FWHM_MM / 2.355

# Pixels above this value are considered useful tissue.
BACKGROUND_THRESHOLD = 0.0

VALID_EXT = ".png"

EXPECTED_SUBJECTS = 902
EXPECTED_PNGS_PER_SUBJECT = 20


# =========================================================
# INPUT AND OUTPUT FOLDERS
# =========================================================

# Folders where PNG images are directly inside the input folder.
# No subfolders are expected here.
FLAT_FOLDER_PAIRS = [
    # Axial crop images
    {
        "name": "axial central crop",
        "input": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_central_crop"),
        "smooth_output": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_central_crop_smooth"),
        "final_output": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_central_crop_final"),
    },
    {
        "name": "axial minus1 crop",
        "input": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_minus1_crop"),
        "smooth_output": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_minus1_crop_smooth"),
        "final_output": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_minus1_crop_final"),
    },
    {
        "name": "axial plus1 crop",
        "input": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_plus1_crop"),
        "smooth_output": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_plus1_crop_smooth"),
        "final_output": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_plus1_crop_final"),
    },

    # Coronal patch images
    {
        "name": "coronal central patch",
        "input": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_central_patch"),
        "smooth_output": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_central_patch_smooth"),
        "final_output": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_central_patch_final"),
    },
    {
        "name": "coronal minus1 patch",
        "input": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_minus1_patch"),
        "smooth_output": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_minus1_patch_smooth"),
        "final_output": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_minus1_patch_final"),
    },
    {
        "name": "coronal plus1 patch",
        "input": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_plus1_patch"),
        "smooth_output": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_plus1_patch_smooth"),
        "final_output": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_plus1_patch_final"),
    },
]

# Folder where PNG images are stored inside subject-level subfolders.
CORONAL_SLICES_CONFIG = {
    "name": "coronal slices resized",
    "input_root": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal_slices_resized"),
    "smooth_output_root": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal_slices_resized_smooth"),
    "final_output_root": Path("/pool/home/AD_Multimodal/ADNI/irene_adni/coronal_slices_resized_final"),
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def load_png_as_float(path):
    """
    Load a PNG image and convert it to float32 in the range [0, 1].
    """

    path = Path(path)

    with Image.open(path) as img:
        # Convert to grayscale if needed.
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
    Save a float image in the range [0, 1] as an 8-bit PNG.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr * 255).round().astype(np.uint8)

    img = Image.fromarray(arr_uint8)
    img.save(path)


def process_image(image_path, smooth_output_dir, final_output_dir):
    """
    Process one image:
    1. Load the original image.
    2. Apply Gaussian smoothing.
    3. Calculate vmin and vmax from the smoothed image, only inside the mask.
    4. Normalize the original image using vmin and vmax.
    5. Save the smooth image and the final normalized image.
    """

    image_path = Path(image_path)
    smooth_output_dir = Path(smooth_output_dir)
    final_output_dir = Path(final_output_dir)

    original = load_png_as_float(image_path)

    # Mask from the original image to exclude black background.
    mask = original > BACKGROUND_THRESHOLD

    if not np.any(mask):
        return False, "no useful pixels found"

    # Gaussian smoothing.
    smooth = gaussian_filter(original, sigma=SIGMA)

    # vmin and vmax from smoothed image, only inside useful tissue.
    vals = smooth[mask]
    vmin = vals.min()
    vmax = vals.max()

    if vmax <= vmin:
        return False, "vmax <= vmin"

    # Normalize original image using vmin and vmax from the smoothed image.
    final = (original - vmin) / (vmax - vmin)
    final = np.clip(final, 0.0, 1.0)

    stem = image_path.stem

    smooth_path = smooth_output_dir / f"{stem}_smooth.png"
    final_path = final_output_dir / f"{stem}_final.png"

    save_float_png(smooth_path, smooth)
    save_float_png(final_path, final)

    return True, None


def get_png_files(input_dir):
    """
    Get original PNG files from a folder, excluding already processed images.
    """

    input_dir = Path(input_dir)

    png_files = sorted([
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == VALID_EXT
        and not p.name.endswith("_smooth.png")
        and not p.name.endswith("_final.png")
    ])

    return png_files


def get_iterator(files, description):
    """
    Return a tqdm iterator if tqdm is available.
    Otherwise, return the original list.
    """

    if tqdm is not None:
        return tqdm(files, desc=description, unit="img")

    return files


# =========================================================
# PROCESS FLAT FOLDERS
# =========================================================

def process_flat_folder(name, input_dir, smooth_output_dir, final_output_dir):
    """
    Process all PNG images directly inside one input folder.
    This function does not search inside subfolders.
    """

    input_dir = Path(input_dir)
    smooth_output_dir = Path(smooth_output_dir)
    final_output_dir = Path(final_output_dir)

    print("\n" + "=" * 70)
    print(f"Processing folder: {name}")
    print("=" * 70)
    print(f"Input folder:         {input_dir}")
    print(f"Smooth output folder: {smooth_output_dir}")
    print(f"Final output folder:  {final_output_dir}")

    if not input_dir.exists():
        print(f"AVISO: no existe la carpeta de entrada: {input_dir}")

        return {
            "name": name,
            "exists": False,
            "n_images": 0,
            "ok_count": 0,
            "skipped": [],
        }

    smooth_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    png_files = get_png_files(input_dir)

    print(f"Number of images:     {len(png_files)}")

    skipped = []
    ok_count = 0

    iterator = get_iterator(png_files, description=input_dir.name)

    for i, image_path in enumerate(iterator, start=1):
        success, reason = process_image(
            image_path=image_path,
            smooth_output_dir=smooth_output_dir,
            final_output_dir=final_output_dir,
        )

        if success:
            ok_count += 1

            if tqdm is None:
                print(f"OK [{i}/{len(png_files)}]: {image_path.name}")

        else:
            skipped.append((image_path.name, reason))

            if tqdm is None:
                print(f"SKIPPED [{i}/{len(png_files)}]: {image_path.name} - {reason}")

    print(f"\nResumen {name}:")
    print(f"Imágenes encontradas: {len(png_files)}")
    print(f"Imágenes procesadas correctamente: {ok_count}")
    print(f"Imágenes saltadas: {len(skipped)}")

    if skipped:
        print("Imágenes saltadas:")
        for image_name, reason in skipped:
            print(f"  {image_name}: {reason}")

    return {
        "name": name,
        "exists": True,
        "n_images": len(png_files),
        "ok_count": ok_count,
        "skipped": skipped,
    }


# =========================================================
# PROCESS SUBJECT-LEVEL FOLDERS
# =========================================================

def process_subject_folder(subject_dir, smooth_output_root, final_output_root):
    """
    Process all PNG images inside one subject folder.
    The same subject folder is created in the output roots.
    """

    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name

    smooth_subject_dir = Path(smooth_output_root) / subject_id
    final_subject_dir = Path(final_output_root) / subject_id

    smooth_subject_dir.mkdir(parents=True, exist_ok=True)
    final_subject_dir.mkdir(parents=True, exist_ok=True)

    png_files = get_png_files(subject_dir)

    if len(png_files) != EXPECTED_PNGS_PER_SUBJECT:
        print(
            f"AVISO sujeto {subject_id}: "
            f"tiene {len(png_files)} PNG, esperado {EXPECTED_PNGS_PER_SUBJECT}."
        )

    skipped = []
    ok_count = 0

    iterator = get_iterator(png_files, description=subject_id)

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


def process_subject_tree(name, input_root, smooth_output_root, final_output_root):
    """
    Process PNG images stored in subject-level folders.
    The subject-level structure is preserved in the output folders.
    """

    input_root = Path(input_root)
    smooth_output_root = Path(smooth_output_root)
    final_output_root = Path(final_output_root)

    print("\n" + "=" * 70)
    print(f"Processing subject-level folder: {name}")
    print("=" * 70)
    print(f"Input root:         {input_root}")
    print(f"Smooth output root: {smooth_output_root}")
    print(f"Final output root:  {final_output_root}")

    if not input_root.exists():
        print(f"AVISO: no existe la carpeta de entrada: {input_root}")

        return {
            "name": name,
            "exists": False,
            "n_subjects": 0,
            "n_images": 0,
            "ok_count": 0,
            "skipped": [],
            "subject_warning_count": 0,
        }

    smooth_output_root.mkdir(parents=True, exist_ok=True)
    final_output_root.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([
        p for p in input_root.iterdir()
        if p.is_dir()
    ])

    print(f"Sujetos encontrados: {len(subject_dirs)}")

    if len(subject_dirs) != EXPECTED_SUBJECTS:
        print(
            f"AVISO: se esperaban {EXPECTED_SUBJECTS} sujetos, "
            f"pero se han encontrado {len(subject_dirs)}."
        )

    total_ok = 0
    total_images = 0
    total_skipped = []
    subject_warning_count = 0

    for subject_index, subject_dir in enumerate(subject_dirs, start=1):
        print("\n" + "-" * 70)
        print(f"Sujeto [{subject_index}/{len(subject_dirs)}]: {subject_dir.name}")
        print("-" * 70)

        ok_count, skipped, n_images = process_subject_folder(
            subject_dir=subject_dir,
            smooth_output_root=smooth_output_root,
            final_output_root=final_output_root,
        )

        total_ok += ok_count
        total_images += n_images

        if n_images != EXPECTED_PNGS_PER_SUBJECT:
            subject_warning_count += 1

        for image_name, reason in skipped:
            total_skipped.append((subject_dir.name, image_name, reason))

    print(f"\nResumen {name}:")
    print(f"Sujetos procesados: {len(subject_dirs)}")
    print(f"Sujetos con número de PNG distinto a {EXPECTED_PNGS_PER_SUBJECT}: {subject_warning_count}")
    print(f"Imágenes encontradas: {total_images}")
    print(f"Imágenes procesadas correctamente: {total_ok}")
    print(f"Imágenes saltadas: {len(total_skipped)}")

    if total_skipped:
        print("Imágenes saltadas:")
        for subject_id, image_name, reason in total_skipped:
            print(f"  {subject_id}/{image_name}: {reason}")

    return {
        "name": name,
        "exists": True,
        "n_subjects": len(subject_dirs),
        "n_images": total_images,
        "ok_count": total_ok,
        "skipped": total_skipped,
        "subject_warning_count": subject_warning_count,
    }


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 70)
    print("ADNI Gaussian normalization")
    print("=" * 70)
    print(f"Using FWHM = {FWHM_MM} mm")
    print(f"Using sigma = {SIGMA:.4f} pixels")

    if tqdm is None:
        print("AVISO: tqdm no está instalado. Se usará progreso con print.")

    summaries = []

    # Process axial crop images and coronal patch images.
    for folders in FLAT_FOLDER_PAIRS:
        summary = process_flat_folder(
            name=folders["name"],
            input_dir=folders["input"],
            smooth_output_dir=folders["smooth_output"],
            final_output_dir=folders["final_output"],
        )
        summaries.append(summary)

    # Process coronal slices stored in subject-level folders.
    coronal_summary = process_subject_tree(
        name=CORONAL_SLICES_CONFIG["name"],
        input_root=CORONAL_SLICES_CONFIG["input_root"],
        smooth_output_root=CORONAL_SLICES_CONFIG["smooth_output_root"],
        final_output_root=CORONAL_SLICES_CONFIG["final_output_root"],
    )
    summaries.append(coronal_summary)

    # Final global summary.
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    total_images = 0
    total_ok = 0
    total_skipped = 0

    for summary in summaries:
        print(f"\n{summary['name']}")
        print(f"Existe: {summary['exists']}")
        print(f"Imágenes encontradas: {summary['n_images']}")
        print(f"Imágenes procesadas correctamente: {summary['ok_count']}")
        print(f"Imágenes saltadas: {len(summary['skipped'])}")

        total_images += summary["n_images"]
        total_ok += summary["ok_count"]
        total_skipped += len(summary["skipped"])

    print("\nTOTAL")
    print(f"Imágenes encontradas: {total_images}")
    print(f"Imágenes procesadas correctamente: {total_ok}")
    print(f"Imágenes saltadas: {total_skipped}")

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()