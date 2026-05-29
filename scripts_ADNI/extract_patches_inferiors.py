#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:47:30 2026

@author: irene
"""

"""
script per fer els patch de les 3 slices centrals vista coronal
patch inferior dret --> 64x64 (pero q no comenci a la cantonada inferior, sino a la zona d'interès)
patch inferior esquerra + flipped --> 64x64 (a la mateixa zona que la dreta)
"""

from pathlib import Path
from PIL import Image, ImageOps

folders = {
    "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_central":
    "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_central_patch",

    "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_minus1":
    "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_minus1_patch",

    "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_plus1":
    "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_plus1_patch",
}

PATCH_SIZE = 64
MARGIN_BOTTOM = 28
MARGIN_SIDE = 12


def get_boxes(width, height, patch_size, margin_bottom, margin_side):
    y1 = height - margin_bottom - patch_size
    y2 = height - margin_bottom

    left_box = (
        margin_side,
        y1,
        margin_side + patch_size,
        y2
    )

    right_box = (
        width - margin_side - patch_size,
        y1,
        width - margin_side,
        y2
    )

    return left_box, right_box


for input_dir, output_dir in folders.items():
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))

    print(f"Procesando: {input_dir}")
    print(f"Imágenes encontradas: {len(png_files)}")

    for img_path in png_files:
        img = Image.open(img_path)

        width, height = img.size

        if width != 182 or height != 182:
            print(f"Atención: {img_path.name} tiene tamaño {width}x{height}")

        left_box, right_box = get_boxes(
            width,
            height,
            PATCH_SIZE,
            MARGIN_BOTTOM,
            MARGIN_SIDE
        )

        left_patch = img.crop(left_box)
        right_patch = img.crop(right_box)

        # Flip horizontal del patch izquierdo
        left_patch_flipped = ImageOps.mirror(left_patch)

        stem = img_path.stem

        right_patch.save(output_dir / f"{stem}_patch.png")
        left_patch_flipped.save(output_dir / f"{stem}_patch_flipped.png")

    print(f"Guardadas {len(png_files) * 2} imágenes en {output_dir}")
    print("-" * 60)