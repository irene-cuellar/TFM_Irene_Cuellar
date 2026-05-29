#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrae las 20 slices centrales de la vista coronal de cada sujeto ADNI.

Salida esperada:
/pool/home/AD_Multimodal/ADNI/irene_adni/coronal_slices/sub-ADNI130S5258/*.png
"""

import os
from glob import glob

import numpy as np
import nibabel as nib
from PIL import Image

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

path_base = '/pool/home/AD_Multimodal/ADNI/irene_adni/anat'
out_base = '/pool/home/AD_Multimodal/ADNI/irene_adni/coronal_slices'

expected_shape = (182, 218, 182)
background_threshold = 0.0

# Número de slices centrales a extraer en vista coronal
n_slices = 20

# Si True, usa el centro fijo de expected_shape.
# Para expected_shape = (182, 218, 182), el centro coronal es 218 // 2 = 109.
use_fixed_center = True

# Si True, procesa solo el primer archivo válido encontrado por sujeto.
# Así cada carpeta de sujeto tendrá exactamente 20 PNG.
# Si quieres 20 slices por cada sesión/run encontrado, ponlo en False.
only_first_file_per_subject = True

os.makedirs(out_base, exist_ok=True)

# =============================================================================
# FUNCIONES
# =============================================================================

def save_png(slice_2d, out_path):
    """Guarda una slice 2D normalizada [0, 1] como PNG en escala de grises."""
    slice_2d = np.clip(slice_2d, 0.0, 1.0)
    arr = (slice_2d * 255).round().astype(np.uint8)
    Image.fromarray(arr, mode='L').save(out_path)


def normalize_volume(data, threshold=0.0):
    """Normaliza el volumen usando solo voxeles por encima del umbral."""
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    mask = data > threshold
    data_norm = np.zeros_like(data, dtype=np.float32)

    if np.any(mask):
        vals = data[mask]
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            data_norm[mask] = (vals - vmin) / (vmax - vmin)

    return data_norm


def extract_coronal_slice(data, idx):
    """Extrae una slice coronal: data[:, idx, :]."""
    sl = data[:, idx, :]
    return np.nan_to_num(np.rot90(sl), nan=0.0)


def get_filepaths(subj_path, subj, ses):
    """Busca los archivos T1w procesados dentro de una sesión."""
    patterns = [
        f'{subj}_{ses}_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz',
        f'{subj}_{ses}_run-*_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz',
        f'{subj}_{ses}_acq-*_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz',
        f'{subj}_{ses}_acq-*_run-*_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz',
    ]

    files = []
    for p in patterns:
        full = os.path.join(subj_path, ses, p)
        if '*' in p:
            files.extend(glob(full))
        elif os.path.exists(full):
            files.append(full)

    return sorted(set(files))


def get_coronal_indices(shape):
    """Devuelve los índices de las n_slices centrales en el eje coronal."""
    if use_fixed_center:
        cor_center = expected_shape[1] // 2
    else:
        cor_center = shape[1] // 2

    half = n_slices // 2

    if n_slices % 2 == 0:
        # Para 20 slices: centro-10 ... centro+9
        start = cor_center - half
        end = cor_center + half
    else:
        # Por si algún día usas un número impar
        start = cor_center - half
        end = cor_center + half + 1

    return list(range(start, end))


def indices_in_bounds(shape, indices):
    """Comprueba que todos los índices coronales caen dentro del eje Y."""
    return all(0 <= idx < shape[1] for idx in indices)

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
n_skipped_all_existing = 0
missing_files = []
wrong_shape = []
out_of_bounds = []
load_errors = []

# =============================================================================
# PROCESADO
# =============================================================================

for subj in subjects:
    subj_path = os.path.join(path_base, subj)
    subj_out_dir = os.path.join(out_base, subj)
    os.makedirs(subj_out_dir, exist_ok=True)

    sessions = sorted([
        d for d in os.listdir(subj_path)
        if d.startswith('ses-') and os.path.isdir(os.path.join(subj_path, d))
    ])

    subject_done = False

    for ses in sessions:
        files_found = get_filepaths(subj_path, subj, ses)

        if not files_found:
            missing_files.append(os.path.join(subj_path, ses))
            continue

        for filepath in files_found:
            try:
                basename = os.path.basename(filepath).replace('.nii.gz', '')

                # Si procesamos solo un archivo por sujeto, los nombres son simples.
                # Si procesamos varios archivos por sujeto, se añade basename para evitar sobrescrituras.
                if only_first_file_per_subject:
                    output_paths = [
                        os.path.join(subj_out_dir, f'{subj}_coronal_slice_{i:02d}.png')
                        for i in range(n_slices)
                    ]
                else:
                    output_paths = [
                        os.path.join(subj_out_dir, f'{basename}_coronal_slice_{i:02d}.png')
                        for i in range(n_slices)
                    ]

                if all(os.path.exists(p) for p in output_paths):
                    n_skipped_all_existing += 1
                    print(f'Ya existen las {n_slices} slices, se salta: {basename}')
                    subject_done = True
                    if only_first_file_per_subject:
                        break
                    else:
                        continue

                nii = nib.as_closest_canonical(nib.load(filepath))

                if expected_shape is not None and nii.shape != expected_shape:
                    wrong_shape.append((subj, ses, os.path.basename(filepath), nii.shape))
                    continue

                data = normalize_volume(
                    nii.get_fdata(),
                    threshold=background_threshold
                )

                indices = get_coronal_indices(data.shape)

                if not indices_in_bounds(data.shape, indices):
                    out_of_bounds.append((subj, ses, os.path.basename(filepath), data.shape, indices))
                    continue

                for i, idx in enumerate(indices):
                    out_path = output_paths[i]
                    if not os.path.exists(out_path):
                        sl = extract_coronal_slice(data, idx)
                        save_png(sl, out_path)
                        saved_slices += 1
                        print(f'Guardado: {out_path}')

                saved_subjects += 1
                subject_done = True

                if only_first_file_per_subject:
                    break

            except Exception as e:
                load_errors.append((subj, ses, os.path.basename(filepath), str(e)))

        if subject_done and only_first_file_per_subject:
            break

# =============================================================================
# RESUMEN
# =============================================================================

print('\n' + '=' * 60)
print('RESUMEN')
print('=' * 60)

print(f'Sujetos con slices guardadas/procesadas: {saved_subjects}')
print(f'Slices nuevas guardadas: {saved_slices}')
print(f'Archivos con las {n_slices} slices ya existentes: {n_skipped_all_existing}')
print(f'Carpetas/sesiones sin archivo encontrado: {len(missing_files)}')
print(f'Shape distinta: {len(wrong_shape)}')
print(f'Fuera de rango: {len(out_of_bounds)}')
print(f'Errores de carga: {len(load_errors)}')

if missing_files:
    print('\nPrimeras carpetas/sesiones sin archivo encontrado:')
    for x in missing_files[:10]:
        print(f'  - {x}')

if wrong_shape:
    print('\nPrimeros casos con shape distinta:')
    for subj, ses, fname, shape in wrong_shape[:10]:
        print(f'  - {subj}, {ses}, {fname}: {shape}')

if out_of_bounds:
    print('\nPrimeros casos fuera de rango:')
    for subj, ses, fname, shape, indices in out_of_bounds[:10]:
        print(f'  - {subj}, {ses}, {fname}: shape={shape}, indices={indices}')

if load_errors:
    print('\nPrimeros errores de carga:')
    for subj, ses, fname, err in load_errors[:10]:
        print(f'  - {subj}, {ses}, {fname}: {err}')
