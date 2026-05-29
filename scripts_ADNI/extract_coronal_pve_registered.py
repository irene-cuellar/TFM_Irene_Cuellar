#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:34:35 2026

@author: irene
"""
"""
Script para extraer la slice coronal central de los mapas PVE registrados
a template 1mm en ADNI.

Para cada sujeto/sesión busca:
- pve_0
- pve_1
- pve_2

Ejemplo:
sub-XXX_ses-YYY_run-02_T1w_fast_seg_pve_0_desc-reg2-template.nii.gz

Los PNG se guardan en:
.../pve_registered

Además:
- si el PNG coronal ya existe, se salta y no se reescribe
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

path_base = '/pool/home/AD_Multimodal/ADNI/irene_adni/anat'

# Carpeta de salida
path_out = '/pool/home/AD_Multimodal/ADNI/irene_adni/pve_registered'
os.makedirs(path_out, exist_ok=True)

# Shape esperada tras registro al template
expected_shape = (182, 218, 182)

# Si True, usa el centro fijo de expected_shape
# Si False, usa el centro real de cada imagen
use_fixed_center = True

# Tipos PVE a extraer
pve_labels = [0, 1, 2]

# =============================================================================
# BUSCAR SUJETOS
# =============================================================================

subjects = sorted([
    d for d in os.listdir(path_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(path_base, d))
])

if len(subjects) == 0:
    raise ValueError(f'No se han encontrado carpetas sub-* en: {path_base}')

print(f'Se han encontrado {len(subjects)} sujetos.')

# =============================================================================
# ÍNDICE CENTRAL FIJO CORONAL
# =============================================================================
# En una imagen 3D con forma (X, Y, Z):
# - axial:   data[:, :, z]
# - coronal: data[:, y, :]
# - sagital: data[x, :, :]

if use_fixed_center:
    if expected_shape is None:
        raise ValueError('Si use_fixed_center=True, expected_shape no puede ser None')
    slice_idx_fixed = expected_shape[1] // 2
    print(f'Índice coronal central fijo: {slice_idx_fixed}')

# =============================================================================
# CONTADORES Y REGISTROS
# =============================================================================

n_saved = 0
n_skipped_existing = 0
missing_files = []
wrong_shape = []
load_errors = []

# =============================================================================
# PROCESAR SUJETOS Y SESIONES
# =============================================================================

for subj in subjects:
    subj_path = os.path.join(path_base, subj)

    sessions = sorted([
        d for d in os.listdir(subj_path)
        if d.startswith('ses-') and os.path.isdir(os.path.join(subj_path, d))
    ])

    if len(sessions) == 0:
        print(f'Aviso: {subj} no tiene carpetas ses-*')
        continue

    for ses in sessions:
        ses_path = os.path.join(subj_path, ses)

        for pve in pve_labels:
            # -----------------------------------------------------------------
            # Buscar posibles archivos PVE con distintos patrones de nombre
            # -----------------------------------------------------------------
            pattern_no_run = os.path.join(
                ses_path,
                f'{subj}_{ses}_T1w_fast_seg_pve_{pve}_desc-reg2-template.nii.gz'
            )

            pattern_run = os.path.join(
                ses_path,
                f'{subj}_{ses}_run-*_T1w_fast_seg_pve_{pve}_desc-reg2-template.nii.gz'
            )

            pattern_acq = os.path.join(
                ses_path,
                f'{subj}_{ses}_acq-*_T1w_fast_seg_pve_{pve}_desc-reg2-template.nii.gz'
            )

            pattern_acq_run = os.path.join(
                ses_path,
                f'{subj}_{ses}_acq-*_run-*_T1w_fast_seg_pve_{pve}_desc-reg2-template.nii.gz'
            )

            files_found = []

            if os.path.exists(pattern_no_run):
                files_found.append(pattern_no_run)

            files_found.extend(glob(pattern_run))
            files_found.extend(glob(pattern_acq))
            files_found.extend(glob(pattern_acq_run))

            # Quitar duplicados y ordenar
            files_found = sorted(set(files_found))

            if len(files_found) == 0:
                missing_files.append((subj, ses, f'pve_{pve}'))
                continue

            # Procesar todos los archivos encontrados de ese PVE
            for filepath in files_found:
                try:
                    basename = os.path.basename(filepath).replace('.nii.gz', '')
                    name_out = f'{basename}_coronal.png'
                    filepath_out = os.path.join(path_out, name_out)

                    # Si ya existe, saltar
                    if os.path.exists(filepath_out):
                        n_skipped_existing += 1
                        print(f'Ya existe, se salta: {name_out}')
                        continue

                    # Cargar NIfTI
                    nii = nib.load(filepath)

                    # Reorientar a canónica
                    nii = nib.as_closest_canonical(nii)

                    # Comprobar shape
                    if expected_shape is not None and nii.shape != expected_shape:
                        wrong_shape.append((subj, ses, os.path.basename(filepath), nii.shape))
                        continue

                    data = nii.get_fdata()
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                    # Los mapas PVE suelen estar en rango 0-1.
                    # Se recorta por seguridad para guardar PNG en escala de grises.
                    data = np.clip(data, 0.0, 1.0)

                    # Slice coronal central
                    if use_fixed_center:
                        slice_idx = slice_idx_fixed
                    else:
                        slice_idx = data.shape[1] // 2

                    slice_2d = data[:, slice_idx, :]

                    # Rotación para visualización consistente
                    slice_2d = np.rot90(slice_2d)
                    slice_2d = np.nan_to_num(slice_2d, nan=0.0)

                    # Guardar PNG
                    plt.imsave(filepath_out, slice_2d, cmap='gray', vmin=0, vmax=1)

                    n_saved += 1
                    print(f'Guardado: {name_out}')

                except Exception as e:
                    load_errors.append((subj, ses, os.path.basename(filepath), str(e)))

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print('\n' + '=' * 60)
print('RESUMEN')
print('=' * 60)
print(f'Slices guardadas nuevas: {n_saved}')
print(f'Slices ya existentes saltadas: {n_skipped_existing}')
print(f'Archivos PVE no encontrados: {len(missing_files)}')
print(f'Sujetos/sesiones con shape distinta: {len(wrong_shape)}')
print(f'Errores de carga: {len(load_errors)}')
print(f'Carpeta de salida: {path_out}')

if len(missing_files) > 0:
    print('\nPrimeros PVE no encontrados:')
    for subj, ses, pve_name in missing_files[:10]:
        print(f'  - {subj}, {ses}, {pve_name}')
    if len(missing_files) > 10:
        print(f'  ... y {len(missing_files) - 10} más')

if len(wrong_shape) > 0:
    print('\nPrimeros sujetos/sesiones con shape distinta:')
    for subj, ses, fname, shape in wrong_shape[:10]:
        print(f'  - {subj}, {ses}, {fname}: {shape}')
    if len(wrong_shape) > 10:
        print(f'  ... y {len(wrong_shape) - 10} más')

if len(load_errors) > 0:
    print('\nPrimeros errores de carga:')
    for subj, ses, fname, err in load_errors[:10]:
        print(f'  - {subj}, {ses}, {fname}: {err}')
    if len(load_errors) > 10:
        print(f'  ... y {len(load_errors) - 10} más')
