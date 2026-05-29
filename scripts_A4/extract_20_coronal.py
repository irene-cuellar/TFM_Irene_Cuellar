#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:13:37 2026

@author: irene
"""

"""
Extrae las 20 slices coronales centrales de cada sujeto.

Entrada esperada:
/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/anat/sub-XXX/ses-01/sub-XXX_ses-01_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz

Salida esperada:
/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/coronal/sub-XXX/*.png
"""

import os
import numpy as np
import nibabel as nib
from PIL import Image

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

path_base = '/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/anat'
out_base = '/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/coronal'

ses_session = 'ses-01'
n_slices = 20
background_threshold = 0.0

os.makedirs(out_base, exist_ok=True)

# =============================================================================
# FUNCIONES
# =============================================================================

def save_png(slice_2d, out_path):
    slice_2d = np.clip(slice_2d, 0.0, 1.0)
    arr = (slice_2d * 255).round().astype(np.uint8)
    Image.fromarray(arr, mode='L').save(out_path)


def normalize_volume(data, threshold=0.0):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    mask = data > threshold
    data_norm = np.zeros_like(data, dtype=np.float32)

    if np.any(mask):
        vals = data[mask]
        vmin = vals.min()
        vmax = vals.max()

        if vmax > vmin:
            data_norm[mask] = (vals - vmin) / (vmax - vmin)

    return data_norm


def extract_coronal_slice(data, idx):
    sl = data[:, idx, :]
    return np.nan_to_num(np.rot90(sl), nan=0.0)


def get_coronal_indices(shape):
    cor_center = shape[1] // 2
    half = n_slices // 2

    start = cor_center - half
    end = cor_center + half

    return list(range(start, end))


def get_input_filepath(subj):
    filename = f'{subj}_{ses_session}_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz'

    filepath = os.path.join(
        path_base,
        subj,
        ses_session,
        filename
    )

    return filepath


def get_output_paths(subj):
    subj_out_dir = os.path.join(out_base, subj)
    os.makedirs(subj_out_dir, exist_ok=True)

    output_paths = [
        os.path.join(
            subj_out_dir,
            f'{subj}_coronal_slice_{i:02d}.png'
        )
        for i in range(n_slices)
    ]

    return output_paths

# =============================================================================
# SUJETOS
# =============================================================================

subjects = sorted([
    d for d in os.listdir(path_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(path_base, d))
])

if not subjects:
    raise ValueError(f'No se han encontrado carpetas sub-* en: {path_base}')

print(f'Se han encontrado {len(subjects)} sujetos.')

# =============================================================================
# CONTADORES
# =============================================================================

saved_subjects = 0
saved_slices = 0
skipped_existing = 0
missing_files = []
load_errors = []

# =============================================================================
# PROCESADO
# =============================================================================

for subj in subjects:
    filepath = get_input_filepath(subj)
    output_paths = get_output_paths(subj)

    if all(os.path.exists(p) for p in output_paths):
        skipped_existing += 1
        print(f'Ya existen las {n_slices} slices, se salta: {subj}')
        continue

    if not os.path.exists(filepath):
        missing_files.append(filepath)
        continue

    try:
        nii = nib.as_closest_canonical(nib.load(filepath))

        data = normalize_volume(
            nii.get_fdata(),
            threshold=background_threshold
        )

        indices = get_coronal_indices(data.shape)

        for i, idx in enumerate(indices):
            out_path = output_paths[i]

            if not os.path.exists(out_path):
                sl = extract_coronal_slice(data, idx)
                save_png(sl, out_path)
                saved_slices += 1
                print(f'Guardado: {out_path}')

        saved_subjects += 1

    except Exception as e:
        load_errors.append((subj, filepath, str(e)))

# =============================================================================
# RESUMEN
# =============================================================================

print('\n' + '=' * 60)
print('RESUMEN')
print('=' * 60)

print(f'Sujetos procesados: {saved_subjects}')
print(f'Slices nuevas guardadas: {saved_slices}')
print(f'Sujetos saltados porque ya tenían las {n_slices} slices: {skipped_existing}')
print(f'Archivos no encontrados: {len(missing_files)}')
print(f'Errores de carga: {len(load_errors)}')

if missing_files:
    print('\nPrimeros archivos no encontrados:')
    for x in missing_files[:10]:
        print(f'  - {x}')

if load_errors:
    print('\nPrimeros errores de carga:')
    for subj, filepath, err in load_errors[:10]:
        print(f'  - {subj}: {filepath}: {err}')