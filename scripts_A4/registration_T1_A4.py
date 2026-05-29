#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:12:31 2026

@author: irene
"""

"""
Registration T1 1mm para A4/LEARN usando HD-BET.

- Busca estrictamente:
  path_base_a4 o path_base_learn / sub-XXX / ses-01 / anat / sub-XXX_ses-01_T1w.nii.gz

- Permite lanzar varios scripts a la vez usando:
  START_IDX y END_IDX
"""

import sys
import os
import shutil
import json
import yaml
import pandas as pd
import time
from contextlib import contextmanager

sys.path.append('/pool/home/AD_Multimodal/Estudio_A4/codes_marina/Scripts_Preproc_Homebrew')

HD_BET_BIN_DIR = '/home/biofisica/anaconda3/bin'
os.environ["PATH"] = HD_BET_BIN_DIR + os.pathsep + os.environ.get("PATH", "")

print("PYTHON:", sys.executable)
print("HD_BET:", shutil.which("hd-bet"))

from logwrapper import log_execution, log_records
from fun_defs import (
    skull_stripping_hd_bet,
    ants_registration_subject_T1_2_template,
    flirt_registration_subject_T1_2_template
)


# =============================================================================
# CONFIG
# =============================================================================

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


config = load_config(
    '/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/config.yaml'
)


# Cambia estos índices para lanzar varios scripts en paralelo
# script 1 -> START_IDX = 0,    END_IDX = 250
# script 2 -> START_IDX = 250,  END_IDX = 500
# script 3 -> START_IDX = 500,  END_IDX = 750
# script 4 -> START_IDX = 750,  END_IDX = 1000
# script 5 -> START_IDX = 1000, END_IDX = 1240

START_IDX = 0
END_IDX = 250


# =============================================================================
# UTILIDADES
# =============================================================================

@contextmanager
def suppress_output():
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


def normalize_group(group_value):
    group_value = str(group_value).strip().lower()

    if group_value == 'a4':
        return 'a4'
    elif group_value == 'learn':
        return 'learn'
    else:
        raise ValueError(f"Grupo no reconocido: {group_value}")


def get_input_anat_path(sub_subject, group, ses_session='ses-01'):
    group = normalize_group(group)

    if group == 'a4':
        base_in = config['path_base_a4']
    else:
        base_in = config['path_base_learn']

    path_in_anat = os.path.join(
        base_in,
        sub_subject,
        ses_session,
        'anat'
    )

    name_anat_nii_in = f'{sub_subject}_{ses_session}_T1w.nii.gz'
    full_input_path = os.path.join(path_in_anat, name_anat_nii_in)

    return path_in_anat, name_anat_nii_in, full_input_path


def get_output_anat_path(sub_subject, ses_session='ses-01'):
    output_root = config['output_folder']

    path_out_anat = os.path.join(
        output_root,
        'anat',
        sub_subject,
        ses_session
    )

    os.makedirs(path_out_anat, exist_ok=True)

    return path_out_anat


def get_final_registered_path(sub_subject, ses_session='ses-01'):
    path_out_anat = get_output_anat_path(sub_subject, ses_session)

    final_name = (
        f'{sub_subject}_{ses_session}_T1w_'
        f'seg-brain-hdbet_desc-reg2-template.nii.gz'
    )

    return os.path.join(path_out_anat, final_name)


# =============================================================================
# PIPELINE
# =============================================================================

@log_execution
def anatomical_preprocessing_pipeline(sub_subject, group, ses_session='ses-01'):
    path_in_anat, name_anat_nii_in, full_input_path = get_input_anat_path(
        sub_subject=sub_subject,
        group=group,
        ses_session=ses_session
    )

    path_out_anat = get_output_anat_path(
        sub_subject=sub_subject,
        ses_session=ses_session
    )

    name_template = config['name_template_t1']
    label_registro = config['set_registration_method']

    with suppress_output():
        name_brain_seg_img_anat_in = skull_stripping_hd_bet(
            path_in_anat + '/',
            name_anat_nii_in,
            path_out_anat + '/'
        )

    brain_base = os.path.basename(
        name_brain_seg_img_anat_in
    ).replace('.nii.gz', '')

    name_t1_reg = os.path.join(
        path_out_anat,
        brain_base + '_desc-reg2-template.nii.gz'
    )

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


# =============================================================================
# MAIN
# =============================================================================

def main():
    df = pd.read_csv(config['subject_list'])
    df = df.iloc[START_IDX:END_IDX].copy()

    total = len(df)

    completed = 0
    skipped_existing = 0
    skipped_missing_t1 = 0
    errors = 0

    last_report_time = time.time()

    print(f"Procesando filas del CSV desde {START_IDX} hasta {END_IDX - 1}")
    print(f"Sujetos en este bloque: {total}")

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        sub_subject = row.name_subject
        group = row.group
        ses_session = 'ses-01'

        try:
            final_output = get_final_registered_path(
                sub_subject=sub_subject,
                ses_session=ses_session
            )

            if os.path.isfile(final_output):
                skipped_existing += 1
                continue

            path_in_anat, name_anat_nii_in, full_input_path = get_input_anat_path(
                sub_subject=sub_subject,
                group=group,
                ses_session=ses_session
            )

            if not os.path.isfile(full_input_path):
                msg = f"No existe la imagen T1: {full_input_path}"

                log_records.append({
                    "subject": sub_subject,
                    "group": group,
                    "session": ses_session,
                    "error": msg
                })

                skipped_missing_t1 += 1
                continue

            anatomical_preprocessing_pipeline(
                sub_subject=sub_subject,
                group=group,
                ses_session=ses_session
            )

            completed += 1

        except Exception as e:
            log_records.append({
                "subject": sub_subject,
                "group": group,
                "session": ses_session,
                "error": str(e)
            })

            errors += 1

            print(
                f"Error processing subject {sub_subject} "
                f"({group}, {ses_session}): {str(e)}"
            )

        finally:
            now = time.time()

            if (now - last_report_time >= 900) or (idx == total):
                print(
                    f"Progreso: {idx}/{total} revisados | "
                    f"hechos ahora: {completed} | "
                    f"ya existían: {skipped_existing} | "
                    f"sin T1: {skipped_missing_t1} | "
                    f"errores: {errors}"
                )

                last_report_time = now


# =============================================================================
# LOG
# =============================================================================

def save_log_records():
    output_path = os.path.join(
        config['output_folder'],
        'log_data_1mm.json'
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as json_file:
        json.dump(log_records, json_file, indent=2)


if __name__ == "__main__":
    main()
    save_log_records()