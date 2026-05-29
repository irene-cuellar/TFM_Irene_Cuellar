#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:55:36 2026

@author: irene
"""
"""
Script para hacer skull stripping + registration de T1 a template 1 mm
Caso ADNI:
- dentro de path_base_adni hay una carpeta por sujeto
- dentro, una carpeta por sesión
- se usa la sesión indicada en la columna 'session' del CSV
- dentro de anat:
    * si existe run-02_T1w.nii.gz, se usa esa
    * si solo existe una T1 tipo _T1w.nii.gz, se usa esa
"""

import sys
sys.path.append('/pool/home/AD_Multimodal/Estudio_A4/codes_marina/Scripts_Preproc_Homebrew')
# creo que sirve el script homebrew que ya use para el Studio A4

import os
import shutil

# para poder usar hd_bet (que está dsiponible en ese enviroment)
HD_BET_BIN_DIR = '/home/biofisica/anaconda3/bin'
os.environ["PATH"] = HD_BET_BIN_DIR + os.pathsep + os.environ.get("PATH", "")

print("PYTHON:", sys.executable)
print("HD_BET:", shutil.which("hd-bet"))

from logwrapper import log_execution, log_records
from fun_defs import (
    skull_stripping_hd_bet,
    skull_stripping_fsl_bet,
    skull_stripping_nipype_bet,
    ants_registration_subject_T1_2_template,
    flirt_registration_subject_T1_2_template
)

import json
import yaml
import glob
import pandas as pd
import time
from contextlib import contextmanager

# para silenciar las salidas de hd_bet
@contextmanager
def suppress_output():
    """
    Silencia stdout y stderr a nivel de sistema.
    Útil cuando una función lanza ejecutables externos como hd-bet.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)

    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull)


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


config = load_config('/pool/home/AD_Multimodal/ADNI/irene_adni/config.yaml')

# que sujetos son procesados en cada script (para lanzar varios a la vez)
# script 1 → START_IDX = 0, END_IDX = 200
# script 2 → START_IDX = 200, END_IDX = 400
# script 3 → START_IDX = 400, END_IDX = 600
# script 4 → START_IDX = 600, END_IDX = 800
# script 5 → START_IDX = 800, END_IDX = 1000

START_IDX = 0
END_IDX = 1000
 
###############################################################################
# FUNCIONES AUXILIARES
###############################################################################

def normalize_session(session_value):
    """
    Asegura que la sesión tenga formato ses-XX.
    Ejemplos:
    - 'ses-01' -> 'ses-01'
    - '01'     -> 'ses-01'
    """
    ses = str(session_value).strip()

    if ses == '' or ses.lower() == 'nan':
        raise ValueError('La columna session está vacía o no es válida.')

    if not ses.startswith('ses-'):
        ses = f'ses-{ses}'

    return ses


def get_input_anat_folder(sub_subject, ses_session):
    """
    Devuelve la carpeta anat de entrada:
    path_base_adni / sub_subject / ses_session / anat
    """
    path_in_anat = os.path.join(
        config['path_base_adni'],
        sub_subject,
        ses_session,
        'anat'
    )
    return path_in_anat


def select_t1_file(path_in_anat):
    """
    Busca T1 en la carpeta anat.

    Reglas:
    1) si hay algún archivo que termina en run-02_T1w.nii.gz -> usar ese
    2) si no, usar el primer archivo que termine en _T1w.nii.gz
    3) si no hay ninguno -> devolver None
    """
    file_pattern = os.path.join(path_in_anat, '*T1w.nii.gz')
    print("Searching for files with pattern:", file_pattern)

    matching_files = sorted(glob.glob(file_pattern))

    if not matching_files:
        return None, None, None

    # Prioridad absoluta a run-02
    run02_files = [
        f for f in matching_files
        if os.path.basename(f).endswith('run-02_T1w.nii.gz')
    ]

    if run02_files:
        selected_file = run02_files[0]
    else:
        selected_file = matching_files[0]

    print("Selected T1 file:", selected_file)

    name_anat_nii_in = os.path.basename(selected_file)
    name_t1_base = name_anat_nii_in.replace('.nii.gz', '')

    return selected_file, name_anat_nii_in, name_t1_base


def get_selected_input_info(sub_subject, ses_session):
    """
    Devuelve:
    - path_in_anat
    - name_anat_nii_in
    - full_input_path
    - name_t1_base
    """
    path_in_anat = get_input_anat_folder(sub_subject, ses_session)

    selected_file, name_anat_nii_in, name_t1_base = select_t1_file(path_in_anat)

    if selected_file is None:
        return path_in_anat, None, None, None

    return path_in_anat, name_anat_nii_in, selected_file, name_t1_base


def get_output_anat_path(sub_subject, ses_session):
    """
    Carpeta de salida.
    Como en config output_folder ya apunta a /anat,
    aquí guarda en:
    output_folder / sub_subject / ses_session
    """
    output_root = config['output_folder']
    path_out_anat = os.path.join(output_root, sub_subject, ses_session)
    os.makedirs(path_out_anat, exist_ok=True)
    return path_out_anat


def get_skullstrip_suffix():
    skull_method = config['skull_strip_method']

    if skull_method == 'hd-bet':
        return 'seg-brain-hdbet'
    elif skull_method == 'bet-nipype':
        return 'seg-brain-npbet'
    elif skull_method == 'bet-fslpy':
        return 'seg-brain-bet'
    else:
        raise ValueError(f"Método de skull stripping no reconocido: {skull_method}")


def get_final_registered_path(sub_subject, ses_session, name_anat_nii_in):
    """
    Construye la ruta final esperada del archivo registrado.
    Importante: usa el nombre REAL del input, para que funcione
    tanto con:
    - sub-XXX_ses-YY_T1w.nii.gz
    - sub-XXX_ses-YY_run-02_T1w.nii.gz
    """
    path_out_anat = get_output_anat_path(sub_subject, ses_session)

    suffix = get_skullstrip_suffix()
    input_base = name_anat_nii_in.replace('.nii.gz', '')

    final_name = f'{input_base}_{suffix}_desc-reg2-template.nii.gz'
    return os.path.join(path_out_anat, final_name)


###############################################################################
# PIPELINE
###############################################################################

@log_execution
def anatomical_preprocessing_pipeline(sub_subject, ses_session):
    path_in_anat, name_anat_nii_in, full_input_path, name_t1_base = get_selected_input_info(
        sub_subject=sub_subject,
        ses_session=ses_session
    )

    if full_input_path is None:
        raise FileNotFoundError(
            f'No se encontró ninguna T1 válida en: {path_in_anat}'
        )

    if not os.path.isfile(full_input_path):
        raise FileNotFoundError(
            f'No existe la imagen anatómica: {full_input_path}'
        )

    path_out_anat = get_output_anat_path(sub_subject, ses_session)

    label_skull_stripping_t1 = config['skull_strip_method']
    label_registro = config['set_registration_method']
    name_template = config['name_template_t1']

    # Skull stripping
    if label_skull_stripping_t1 == 'hd-bet':
        with suppress_output():
            name_brain_seg_img_anat_in = skull_stripping_hd_bet(
                path_in_anat + '/',
                name_anat_nii_in,
                path_out_anat + '/'
        )

    elif label_skull_stripping_t1 == 'bet-nipype':
        name_brain_seg_img_anat_in = skull_stripping_nipype_bet(
            path_in_anat + '/',
            name_anat_nii_in,
            path_out_anat + '/'
        )

    elif label_skull_stripping_t1 == 'bet-fslpy':
        name_brain_seg_img_anat_in = skull_stripping_fsl_bet(
            path_in_anat + '/',
            name_anat_nii_in,
            path_out_anat + '/',
            0.7
        )

    else:
        raise ValueError(
            f"Método de skull stripping no reconocido: {label_skull_stripping_t1}"
        )

    # Nombre de salida del registro
    brain_base = os.path.basename(name_brain_seg_img_anat_in).replace('.nii.gz', '')
    name_t1_reg = os.path.join(
        path_out_anat,
        brain_base + '_desc-reg2-template.nii.gz'
    )

    # Registration T1 -> template
    if label_registro == 'ants':
        ants_registration_subject_T1_2_template(
            name_template=name_template,
            name_mov_t1=name_brain_seg_img_anat_in,
            name_t1_reg=name_t1_reg
        )

    elif label_registro == 'flirt':
        name_transf_t1_2_template = os.path.join(
            path_out_anat,
            brain_base + '_desc-reg2-template.mat'
        )

        flirt_registration_subject_T1_2_template(
            name_template=name_template,
            name_mov_t1=name_brain_seg_img_anat_in,
            name_t1_reg=name_t1_reg,
            name_transf_t1_2_template=name_transf_t1_2_template
        )

    else:
        raise ValueError(f"Método de registro no reconocido: {label_registro}")

    return name_t1_reg


###############################################################################
# MAIN
###############################################################################

def main():
    df = pd.read_csv(config['subject_list'])
    df = df.iloc[START_IDX:END_IDX].copy()
    total = len(df)

    print(f"Procesando filas del CSV desde {START_IDX} hasta {END_IDX - 1}")

    completed = 0
    skipped_existing = 0
    skipped_missing_t1 = 0
    errors = 0

    last_report_time = time.time()

    print(f"Sujetos en CSV para procesar: {total}")

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        # Asumo que el CSV tiene columnas:
        # - subject
        # - session
        sub_subject = row.subject
        ses_session = normalize_session(row.session)

        try:
            path_in_anat, name_anat_nii_in, full_input_path, name_t1_base = get_selected_input_info(
                sub_subject=sub_subject,
                ses_session=ses_session
            )

            # Si no hay T1 válida, se salta
            if full_input_path is None:
                msg = f'No se encontró ninguna T1 válida en: {path_in_anat}'
                log_records.append({
                    "subject": sub_subject,
                    "session": ses_session,
                    "error": msg
                })
                skipped_missing_t1 += 1
                continue

            # Ruta final esperada de salida
            final_output = get_final_registered_path(
                sub_subject=sub_subject,
                ses_session=ses_session,
                name_anat_nii_in=name_anat_nii_in
            )

            # Si ya existe, se salta
            if os.path.isfile(final_output):
                skipped_existing += 1
                continue

            # Si por cualquier motivo el archivo seleccionado ya no existe
            if not os.path.isfile(full_input_path):
                msg = f'No existe la imagen T1: {full_input_path}'
                log_records.append({
                    "subject": sub_subject,
                    "session": ses_session,
                    "error": msg
                })
                skipped_missing_t1 += 1
                continue

            anatomical_preprocessing_pipeline(
                sub_subject=sub_subject,
                ses_session=ses_session
            )

            completed += 1

        except Exception as e:
            log_records.append({
                "subject": sub_subject,
                "session": ses_session,
                "error": str(e)
            })
            errors += 1
            print(f"Error processing subject {sub_subject} ({ses_session}): {str(e)}")

        finally:
            now = time.time()

            # Informe cada 15 min o al final
            if (now - last_report_time >= 900) or (idx == total):
                print(
                    f"Progreso: {idx}/{total} revisados | "
                    f"hechos ahora: {completed} | "
                    f"ya existían: {skipped_existing} | "
                    f"sin T1: {skipped_missing_t1} | "
                    f"errores: {errors}"
                )
                last_report_time = now


###############################################################################
# LOG
###############################################################################

def save_log_records():
    output_path = os.path.join(config['output_folder'], 'log_data_1mm.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as json_file:
        json.dump(log_records, json_file, indent=2)


if __name__ == "__main__":
    main()
    save_log_records()