#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registration de mapas PVE a template 1 mm usando transforms ya existentes.

IMPORTANTE:
- NO recalcula la registration de la T1.
- Reutiliza los transforms generados previamente por ants.registration(fixed=template, moving=T1).
- Asume que pve_0, pve_1 y pve_2 están en el mismo espacio nativo que la T1 usada para calcular esos transforms.

Pipeline:
1. Lee subject, session y source_dataset del CSV.
2. Busca pve_0, pve_1 y pve_2 en:
   path_base_adni_A/B/C / subject / session / segmentation
3. Busca los transforms ya existentes de T1 nativa -> template en:
   output_folder / subject / session
4. Aplica esos transforms a pve_0, pve_1 y pve_2 con interpolación Linear.
5. Guarda las PVEs registradas en:
   output_folder / subject / session
"""

import os
import glob
import json
import yaml
import time
import pandas as pd
import ants


###############################################################################
# CONFIG
###############################################################################

CONFIG_FILE = "/pool/home/AD_Multimodal/ADNI/irene_adni/config_pve.yaml"


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


config = load_config(CONFIG_FILE)


###############################################################################
# AUXILIARES
###############################################################################

def get_dataset_base_path(source_dataset):
    """
    Devuelve el path base según source_dataset = A, B o C.
    """
    ds = str(source_dataset).strip().upper()
    key = f"path_base_adni_{ds}"

    if key not in config:
        raise ValueError(
            f"source_dataset='{source_dataset}' no reconocido. "
            f"Esperaba A, B o C."
        )

    return config[key]


def get_input_segmentation_folder(sub_subject, ses_session, source_dataset):
    """
    Carpeta donde están las PVE:
    path_base_adni_A/B/C / subject / session / segmentation
    """
    base_path = get_dataset_base_path(source_dataset)

    return os.path.join(
        base_path,
        sub_subject,
        ses_session,
        "segmentation"
    )


def get_output_subject_session_folder(sub_subject, ses_session):
    """
    Carpeta de salida:
    output_folder / subject / session
    """
    path_out = os.path.join(
        config["output_folder"],
        sub_subject,
        ses_session
    )

    os.makedirs(path_out, exist_ok=True)
    return path_out


def find_pve_file(segmentation_folder, sub_subject, ses_session, pve_idx):
    """
    Busca una PVE concreta.

    Nombre esperado:
    sub-XXX_ses-YYY_T1w_fast_seg_pve_0.nii.gz
    """
    expected_name = (
        f"{sub_subject}_{ses_session}_T1w_fast_seg_pve_{pve_idx}.nii.gz"
    )

    expected_path = os.path.join(segmentation_folder, expected_name)

    if os.path.isfile(expected_path):
        return expected_path

    # Fallback por si hubiera alguna variación mínima en el nombre
    pattern = os.path.join(
        segmentation_folder,
        f"{sub_subject}_{ses_session}*pve_{pve_idx}.nii.gz"
    )

    matches = sorted(glob.glob(pattern))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Hay más de una PVE_{pve_idx} posible en {segmentation_folder}: "
            f"{matches}"
        )

    return None


def find_all_pve_files(segmentation_folder, sub_subject, ses_session):
    """
    Devuelve diccionario:
    {
        0: path_pve0,
        1: path_pve1,
        2: path_pve2
    }
    """
    pve_files = {}

    for pve_idx in [0, 1, 2]:
        pve_path = find_pve_file(
            segmentation_folder=segmentation_folder,
            sub_subject=sub_subject,
            ses_session=ses_session,
            pve_idx=pve_idx
        )

        if pve_path is None:
            raise FileNotFoundError(
                f"No se encontró pve_{pve_idx} en {segmentation_folder}"
            )

        pve_files[pve_idx] = pve_path

    return pve_files


def get_registered_pve_output_path(pve_path, output_folder):
    """
    Mantiene la nomenclatura del archivo original y añade:
    _desc-reg2-template.nii.gz
    """
    pve_name = os.path.basename(pve_path)
    pve_base = pve_name.replace(".nii.gz", "")

    out_name = f"{pve_base}_desc-reg2-template.nii.gz"

    return os.path.join(output_folder, out_name)


def require_single_file(matches, description, pattern):
    """
    Exige encontrar exactamente un archivo.
    """
    matches = sorted(set(matches))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No se encontró {description}.\n"
            f"Patrón usado:\n{pattern}"
        )

    if len(matches) > 1:
        msg = f"Se encontró más de un archivo para {description}:\n"
        for path in matches:
            msg += f"  - {path}\n"
        msg += "No elijo automáticamente para evitar aplicar transforms incorrectos."
        raise RuntimeError(msg)

    return matches[0]


def find_existing_t1_to_template_transforms(output_folder, sub_subject, ses_session):
    """
    Busca los transforms YA EXISTENTES de T1 nativa -> template.

    Según el script original:

        ants.registration(
            fixed=template,
            moving=t1_native,
            type_of_transform="SyN"
        )

    Los forward transforms llevan de espacio nativo T1 a espacio template.
    Como las PVE están en el mismo espacio nativo que la T1, usamos:

        *_transform0.nii.gz  -> warp/deformation field forward
        *_transform1.mat     -> affine forward

    Para ants.apply_transforms:

        transformlist = [transform0.nii.gz, transform1.mat]
        whichtoinvert = [False, False]
    """
    prefix = os.path.join(
        output_folder,
        f"{sub_subject}_{ses_session}*T1w_seg-brain-hdbet_desc-reg2-template"
    )

    warp_pattern = prefix + "_transform0.nii.gz"
    affine_pattern = prefix + "_transform1.mat"

    warp_path = require_single_file(
        matches=glob.glob(warp_pattern),
        description="warp forward *_transform0.nii.gz",
        pattern=warp_pattern
    )

    affine_path = require_single_file(
        matches=glob.glob(affine_pattern),
        description="affine forward *_transform1.mat",
        pattern=affine_pattern
    )

    transform_list = [warp_path, affine_path]
    whichtoinvert = [False, False]

    print("Transforms existentes encontrados para T1 native -> template:")
    print("WARP forward:", warp_path)
    print("AFFINE forward:", affine_path)
    print("transformlist:", transform_list)
    print("whichtoinvert:", whichtoinvert)

    return transform_list, whichtoinvert


def check_ants_available():
    """
    Comprueba que ANTsPy está disponible.
    """
    print("ANTsPy importado correctamente.")
    print("ants.__file__:", ants.__file__)

    ants_version = getattr(ants, "__version__", "unknown")
    print("ANTsPy version:", ants_version)


def apply_ants_transforms_to_pve(
    fixed_template,
    moving_pve,
    output_pve,
    transform_list,
    whichtoinvert,
    interpolation="linear"
):
    """
    Aplica la transformación T1 native -> template a una PVE usando ANTsPy.

    Para mapas de probabilidades usamos interpolación linear.
    """
    print("Aplicando transforms a:")
    print(moving_pve)

    fixed = ants.image_read(fixed_template)
    moving = ants.image_read(moving_pve)

    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transform_list,
        whichtoinvert=whichtoinvert,
        interpolator=interpolation
    )

    ants.image_write(warped, output_pve)

    if not os.path.isfile(output_pve):
        raise FileNotFoundError(f"No se generó salida registrada: {output_pve}")

    print("Guardado:")
    print(output_pve)


###############################################################################
# PIPELINE
###############################################################################

def process_subject_session(row):
    sub_subject = str(row["subject"]).strip()
    ses_session = str(row["session"]).strip()
    source_dataset = str(row["source_dataset"]).strip().upper()

    segmentation_folder = get_input_segmentation_folder(
        sub_subject=sub_subject,
        ses_session=ses_session,
        source_dataset=source_dataset
    )

    output_folder = get_output_subject_session_folder(
        sub_subject=sub_subject,
        ses_session=ses_session
    )

    pve_files = find_all_pve_files(
        segmentation_folder=segmentation_folder,
        sub_subject=sub_subject,
        ses_session=ses_session
    )

    pve_outputs = {
        idx: get_registered_pve_output_path(pve_path, output_folder)
        for idx, pve_path in pve_files.items()
    }

    force = bool(config.get("force", False))

    # Si ya existen las 3 PVEs registradas, saltar
    if not force and all(os.path.isfile(path) for path in pve_outputs.values()):
        return {
            "subject": sub_subject,
            "session": ses_session,
            "source_dataset": source_dataset,
            "status": "skipped_existing_all_3_pves",
            "outputs": pve_outputs
        }

    fixed_template = config["name_template_t1"]

    if not os.path.isfile(fixed_template):
        raise FileNotFoundError(f"No existe el template: {fixed_template}")

    transform_list, whichtoinvert = find_existing_t1_to_template_transforms(
        output_folder=output_folder,
        sub_subject=sub_subject,
        ses_session=ses_session
    )

    processed_outputs = {}

    for pve_idx in [0, 1, 2]:
        moving_pve = pve_files[pve_idx]
        output_pve = pve_outputs[pve_idx]

        if not force and os.path.isfile(output_pve):
            print(f"Ya existe PVE_{pve_idx} registrada. Se salta: {output_pve}")
            processed_outputs[pve_idx] = output_pve
            continue

        apply_ants_transforms_to_pve(
            fixed_template=fixed_template,
            moving_pve=moving_pve,
            output_pve=output_pve,
            transform_list=transform_list,
            whichtoinvert=whichtoinvert,
            interpolation="linear"
        )

        processed_outputs[pve_idx] = output_pve

    return {
        "subject": sub_subject,
        "session": ses_session,
        "source_dataset": source_dataset,
        "status": "processed",
        "template": fixed_template,
        "transform_list": transform_list,
        "whichtoinvert": whichtoinvert,
        "outputs": processed_outputs
    }


###############################################################################
# MAIN
###############################################################################

def main():
    check_ants_available()

    df = pd.read_csv(config["subject_list"])

    required_columns = {"subject", "session", "source_dataset"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en el CSV: {missing_columns}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    start_idx = int(config.get("start_idx", 0))
    end_idx = config.get("end_idx", len(df))

    if end_idx is None:
        end_idx = len(df)

    end_idx = int(end_idx)

    df = df.iloc[start_idx:end_idx].copy()

    print(f"Procesando filas CSV desde {start_idx} hasta {end_idx - 1}")
    print(f"Total filas a revisar: {len(df)}")

    log_records = []

    completed = 0
    skipped_existing = 0
    errors = 0

    last_report_time = time.time()

    for local_idx, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            result = process_subject_session(row)

            log_records.append(result)

            if result["status"] == "processed":
                completed += 1
            elif result["status"] == "skipped_existing_all_3_pves":
                skipped_existing += 1

        except Exception as e:
            errors += 1

            sub_subject = str(row.get("subject", "UNKNOWN")).strip()
            ses_session = str(row.get("session", "UNKNOWN")).strip()
            source_dataset = str(row.get("source_dataset", "UNKNOWN")).strip()

            error_record = {
                "subject": sub_subject,
                "session": ses_session,
                "source_dataset": source_dataset,
                "status": "error",
                "error": str(e)
            }

            log_records.append(error_record)

            print(
                f"ERROR | subject={sub_subject} | "
                f"session={ses_session} | "
                f"source_dataset={source_dataset} | "
                f"{str(e)}"
            )

        finally:
            now = time.time()

            if (now - last_report_time >= 900) or (local_idx == len(df)):
                print(
                    f"Progreso: {local_idx}/{len(df)} revisados | "
                    f"procesados: {completed} | "
                    f"ya existían: {skipped_existing} | "
                    f"errores: {errors}"
                )
                last_report_time = now

    log_path = os.path.join(
        config["output_folder"],
        "log_pve_registration_1mm_existing_t1_transforms.json"
    )

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w") as f:
        json.dump(log_records, f, indent=2)

    print("Log guardado en:")
    print(log_path)

    print("Resumen final:")
    print(f"Procesados:   {completed}")
    print(f"Ya existían:  {skipped_existing}")
    print(f"Errores:      {errors}")


if __name__ == "__main__":
    main()
