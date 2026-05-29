#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:11:45 2026

@author: irene
"""
"""
Script para extraer la 3 slices axial centrales de todas las T1 registradas
a template 1mm en ADNI, recorriendo sujetos y sesiones.

Incluye estos casos de nombre:
- sub-XXX_ses-YYY_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz
- sub-XXX_ses-YYY_run-02_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz
- sub-XXX_ses-YYY_acq-1_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz
- sub-XXX_ses-YYY_acq-1_run-02_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz

Además:
- si el PNG axial ya existe, se salta y no se reescribe
"""

import os
import numpy as np
import nibabel as nib
from glob import glob
from PIL import Image

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

path_base = '/pool/home/AD_Multimodal/ADNI/irene_adni/anat'

out_dirs = {
    'axial': {
        'central': '/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_central',
        'minus1':  '/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_minus1',
        'plus1':   '/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_plus1',
    }
}

for view in out_dirs:
    for pos in out_dirs[view]:
        os.makedirs(out_dirs[view][pos], exist_ok=True)

expected_shape = (182, 218, 182)
background_threshold = 0.0
use_fixed_center = True
offset = 1

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
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            data_norm[mask] = (vals - vmin) / (vmax - vmin)

    return data_norm

def extract_slice(data, view, idx):
    if view == 'axial':
        sl = data[:, :, idx]
    else:
        raise ValueError("view debe ser 'axial'")

    return np.nan_to_num(np.rot90(sl), nan=0.0)

def get_filepaths(subj_path, subj, ses):
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

def get_indices(shape):
    if use_fixed_center:
        axial_center = expected_shape[2] // 2
    else:
        axial_center = shape[2] // 2

    return {
        'axial': {
            'minus1': axial_center - offset,
            'central': axial_center,
            'plus1': axial_center + offset
        }
    }

def check_bounds(shape, indices):
    axial_ok = all(
        0 <= indices['axial'][k] < shape[2]
        for k in indices['axial']
    )
    return axial_ok

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

saved = {
    'axial': {'central': 0, 'minus1': 0, 'plus1': 0}
}

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

    sessions = sorted([
        d for d in os.listdir(subj_path)
        if d.startswith('ses-') and os.path.isdir(os.path.join(subj_path, d))
    ])

    for ses in sessions:
        files_found = get_filepaths(subj_path, subj, ses)

        if not files_found:
            missing_files.append(os.path.join(subj_path, ses))
            continue

        for filepath in files_found:
            try:
                basename = os.path.basename(filepath).replace('.nii.gz', '')

                output_paths = {
                    view: {
                        pos: os.path.join(
                            out_dirs[view][pos],
                            f'{basename}_{view}.png'
                        )
                        for pos in ['minus1', 'central', 'plus1']
                    }
                    for view in ['axial']
                }

                if all(
                    os.path.exists(p)
                    for view in output_paths
                    for p in output_paths[view].values()
                ):
                    n_skipped_all_existing += 1
                    print(f'Ya existen las 3 slices axiales, se salta: {basename}')
                    continue

                nii = nib.as_closest_canonical(nib.load(filepath))

                if expected_shape is not None and nii.shape != expected_shape:
                    wrong_shape.append((subj, ses, os.path.basename(filepath), nii.shape))
                    continue

                data = normalize_volume(
                    nii.get_fdata(),
                    threshold=background_threshold
                )

                indices = get_indices(data.shape)

                if not check_bounds(data.shape, indices):
                    out_of_bounds.append((subj, ses, os.path.basename(filepath), data.shape))
                    continue

                for view in ['axial']:
                    for pos in ['minus1', 'central', 'plus1']:
                        out_path = output_paths[view][pos]

                        if not os.path.exists(out_path):
                            sl = extract_slice(data, view, indices[view][pos])
                            save_png(sl, out_path)
                            saved[view][pos] += 1
                            print(f'Guardado {view}_{pos}: {os.path.basename(out_path)}')

            except Exception as e:
                load_errors.append((subj, ses, os.path.basename(filepath), str(e)))

# =============================================================================
# RESUMEN
# =============================================================================

print('\n' + '=' * 60)
print('RESUMEN')
print('=' * 60)

print('\nAXIAL')
print(f"central: {saved['axial']['central']}")
print(f"minus1:  {saved['axial']['minus1']}")
print(f"plus1:   {saved['axial']['plus1']}")

print(f'\nArchivos con las 3 slices axiales ya existentes: {n_skipped_all_existing}')
print(f'Archivos no encontrados: {len(missing_files)}')
print(f'Shape distinta: {len(wrong_shape)}')
print(f'Fuera de rango: {len(out_of_bounds)}')
print(f'Errores de carga: {len(load_errors)}')

if missing_files:
    print('\nPrimeros archivos no encontrados:')
    for x in missing_files[:10]:
        print(f'  - {x}')

if wrong_shape:
    print('\nPrimeros casos con shape distinta:')
    for subj, ses, fname, shape in wrong_shape[:10]:
        print(f'  - {subj}, {ses}, {fname}: {shape}')

if out_of_bounds:
    print('\nPrimeros casos fuera de rango:')
    for subj, ses, fname, shape in out_of_bounds[:10]:
        print(f'  - {subj}, {ses}, {fname}: {shape}')

if load_errors:
    print('\nPrimeros errores de carga:')
    for subj, ses, fname, err in load_errors[:10]:
        print(f'  - {subj}, {ses}, {fname}: {err}')