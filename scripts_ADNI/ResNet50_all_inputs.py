#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:14:40 2026

@author: irene
"""

"""
This script trains a ResNet50 model to classify young CU and old AD subjects
using MRI-derived image inputs. It creates fixed subject-level train, validation
and test splits, trains the model, evaluates performance at subject level, and
estimates the standard deviation of the metrics using bootstrap resampling of
the subjects.
"""

# =========================================================
# IMPORTS
# =========================================================
import os
import re
import gc
import random

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
CSV_PATH = "/pool/home/AD_Multimodal/ADNI/irene_adni/subjects_tfm_irene_filtered_all.csv"
RESULTS_ROOT = "/pool/home/AD_Multimodal/ADNI/irene_adni/results_finals"

SUBJECT_COL = "subject"
AGE_COL = "AGE"
DIAGNOSIS_COL = "DX_binary"

CONTROL_DIAGNOSIS_VALUES = ["CU"]
AD_DIAGNOSIS_VALUES = ["AD"]

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 1e-6
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE_FROM_TRAIN = 0.20
MIN_SUBJECTS_PER_CLASS_FOR_SPLIT = 5

N_BOOTSTRAP = 2000


# =========================================================
# SELECCIONA SOLO UNA OPCIÓN
# =========================================================
# ACTIVE_INPUT_MODE = "axial_1slice"
# ACTIVE_INPUT_MODE = "axial_3slices"
# ACTIVE_INPUT_MODE = "coronal_1slice"
ACTIVE_INPUT_MODE = "coronal_3slices"
# ACTIVE_INPUT_MODE = "coronal_20slices"
# ACTIVE_INPUT_MODE = "coronal_patches"
# ACTIVE_INPUT_MODE = "pve"


INPUT_CONFIGS = {
    "axial_1slice": {
        "kind": "single_slice",
        "title": "Axial 1 slice",
        "paths": {
            "center": "/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_central_crop_final",
        },
    },
    "axial_3slices": {
        "kind": "triplet",
        "title": "Axial 3 slices",
        "paths": {
            "minus1": "/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_minus1_crop_final",
            "center": "/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_central_crop_final",
            "plus1": "/pool/home/AD_Multimodal/ADNI/irene_adni/axial/axial_plus1_crop_final",
        },
    },
    "coronal_1slice": {
        "kind": "single_slice",
        "title": "Coronal 1 slice",
        "paths": {
            "center": "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_central_crop_final",
        },
    },
    "coronal_3slices": {
        "kind": "triplet",
        "title": "Coronal 3 slices",
        "paths": {
            "minus1": "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_minus1_crop_final",
            "center": "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_central_crop_final",
            "plus1": "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_plus1_crop_final",
        },
    },
    "coronal_20slices": {
        "kind": "slice_level",
        "title": "Coronal 20 slices",
        "paths": {
            "root": "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal_slices_resized_final",
        },
    },
    "coronal_patches": {
        "kind": "patch_triplet_aug",
        "title": "Coronal patches",
        "paths": {
            "minus1": "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_minus1_patch_final",
            "center": "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_central_patch_final",
            "plus1": "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_plus1_patch_final",
        },
    },
    "pve": {
        "kind": "pve_channels",
        "title": "PVE channels",
        "paths": {
            "root": "/pool/home/AD_Multimodal/ADNI/irene_adni/pve_registered/resized",
        },
    },
}


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def extraer_subject_id(nombre):
    match = re.search(r"(sub-[A-Za-z0-9]+)", str(nombre))
    return match.group(1) if match else None


def normalizar_subject_id(x):
    if pd.isna(x):
        return None
    return str(x).strip()


def normalizar_diagnosis(x):
    if pd.isna(x):
        return None
    return str(x).strip().upper()


def sorted_pngs(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".png")
    )


def choose_first(paths):
    paths = sorted(paths)
    return paths[0] if len(paths) > 0 else None


def get_slice_index(path):
    name = os.path.basename(path)
    match = re.search(r"slice[_-](\d+)", name)
    return int(match.group(1)) if match else 9999


def get_pve_index(path):
    name = os.path.basename(path)
    match = re.search(r"pve[_-]([012])", name)
    return int(match.group(1)) if match else None


def get_patch_variant(path):
    name = os.path.basename(path).lower()
    return "flipped" if "flipped" in name else "original"


def safe_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan


def safe_nanstd(values, ddof=1):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) <= ddof:
        return np.nan

    return float(np.std(values, ddof=ddof))


def safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def safe_nanpercentile(values, percentile):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan

    return float(np.percentile(values, percentile))


# =========================================================
# METADATA, THRESHOLD Y CASO
# =========================================================
def read_subject_metadata():
    df_csv = pd.read_csv(CSV_PATH, low_memory=False)

    required_cols = [SUBJECT_COL, AGE_COL, DIAGNOSIS_COL]
    missing_cols = [col for col in required_cols if col not in df_csv.columns]

    if missing_cols:
        raise ValueError(f"Faltan estas columnas en el CSV: {missing_cols}")

    df = df_csv[required_cols].copy()
    df["subject_id"] = df[SUBJECT_COL].apply(normalizar_subject_id)
    df["age"] = pd.to_numeric(df[AGE_COL], errors="coerce")
    df["diagnosis_name"] = df[DIAGNOSIS_COL].apply(normalizar_diagnosis)

    df = df.dropna(subset=["subject_id", "age", "diagnosis_name"])
    df = df.drop_duplicates(subset=["subject_id"])
    df = df[["subject_id", "age", "diagnosis_name"]].sort_values("subject_id")

    return df.reset_index(drop=True)


def prepare_young_cu_vs_old_ad_from_csv(df_metadata, age_threshold):
    df = df_metadata.copy()
    df["age_group_name"] = np.where(df["age"] <= age_threshold, "young", "old")

    control_values = set([v.upper() for v in CONTROL_DIAGNOSIS_VALUES])
    ad_values = set([v.upper() for v in AD_DIAGNOSIS_VALUES])

    mask_young_control = (
        df["diagnosis_name"].isin(control_values)
        & (df["age_group_name"] == "young")
    )

    mask_old_ad = (
        df["diagnosis_name"].isin(ad_values)
        & (df["age_group_name"] == "old")
    )

    df_young_control = df[mask_young_control].copy()
    df_young_control["label"] = 0
    df_young_control["target_name"] = "young_CU"

    df_old_ad = df[mask_old_ad].copy()
    df_old_ad["label"] = 1
    df_old_ad["target_name"] = "old_AD"

    df_case = pd.concat([df_young_control, df_old_ad], ignore_index=True)
    df_case = df_case.sort_values("subject_id").reset_index(drop=True)

    return df_case


def check_binary_split_possible(subjects_df):
    counts = subjects_df["label"].value_counts().sort_index()

    if len(counts) != 2:
        raise ValueError(f"Solo hay {len(counts)} clase(s). Counts={counts.to_dict()}")

    if counts.min() < MIN_SUBJECTS_PER_CLASS_FOR_SPLIT:
        raise ValueError(
            f"Hay menos de {MIN_SUBJECTS_PER_CLASS_FOR_SPLIT} sujetos en alguna clase. "
            f"Counts={counts.to_dict()}"
        )


def make_fixed_subject_splits(df_case):
    subjects_df = df_case[
        [
            "subject_id",
            "age",
            "age_group_name",
            "diagnosis_name",
            "label",
            "target_name",
        ]
    ].drop_duplicates("subject_id")

    subjects_df = subjects_df.sort_values("subject_id").reset_index(drop=True)

    check_binary_split_possible(subjects_df)

    train_full_subjects, test_subjects = train_test_split(
        subjects_df,
        test_size=TEST_SIZE,
        stratify=subjects_df["label"],
        random_state=RANDOM_STATE,
    )

    train_subjects, val_subjects = train_test_split(
        train_full_subjects,
        test_size=VAL_SIZE_FROM_TRAIN,
        stratify=train_full_subjects["label"],
        random_state=RANDOM_STATE,
    )

    split_df = pd.concat(
        [
            train_subjects.assign(split="train"),
            val_subjects.assign(split="val"),
            test_subjects.assign(split="test"),
        ],
        ignore_index=True,
    )

    split_df = split_df.sort_values(["split", "subject_id"]).reset_index(drop=True)

    train_ids = set(split_df.loc[split_df["split"] == "train", "subject_id"])
    val_ids = set(split_df.loc[split_df["split"] == "val", "subject_id"])
    test_ids = set(split_df.loc[split_df["split"] == "test", "subject_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    return split_df


# =========================================================
# CREACIÓN DE DATAFRAMES DE IMÁGENES
# =========================================================
def build_subject_file_map(folder):
    mapping = {}

    for path in sorted_pngs(folder):
        subject_id = extraer_subject_id(os.path.basename(path))

        if subject_id is None:
            continue

        mapping.setdefault(subject_id, []).append(path)

    return mapping


def build_single_slice_samples(config):
    file_map = build_subject_file_map(config["paths"]["center"])
    rows = []

    for subject_id, paths in sorted(file_map.items()):
        path = choose_first(paths)

        rows.append(
            {
                "subject_id": subject_id,
                "sample_id": f"{subject_id}_central",
                "input_variant": "central_duplicated",
                "path_ch0": path,
                "path_ch1": path,
                "path_ch2": path,
            }
        )

    return pd.DataFrame(rows)


def build_triplet_samples(config):
    minus_map = build_subject_file_map(config["paths"]["minus1"])
    center_map = build_subject_file_map(config["paths"]["center"])
    plus_map = build_subject_file_map(config["paths"]["plus1"])

    common_subjects = sorted(set(minus_map) & set(center_map) & set(plus_map))
    rows = []

    for subject_id in common_subjects:
        path_m1 = choose_first(minus_map[subject_id])
        path_c = choose_first(center_map[subject_id])
        path_p1 = choose_first(plus_map[subject_id])

        rows.append(
            {
                "subject_id": subject_id,
                "sample_id": f"{subject_id}_triplet",
                "input_variant": "minus1_center_plus1",
                "path_ch0": path_m1,
                "path_ch1": path_c,
                "path_ch2": path_p1,
            }
        )

    return pd.DataFrame(rows)


def build_slice_level_samples(config):
    root = config["paths"]["root"]
    rows = []

    for folder_name in sorted(os.listdir(root)):
        folder_path = os.path.join(root, folder_name)

        if not os.path.isdir(folder_path):
            continue

        subject_id = extraer_subject_id(folder_name)
        pngs = sorted_pngs(folder_path)

        if subject_id is None and len(pngs) > 0:
            subject_id = extraer_subject_id(os.path.basename(pngs[0]))

        if subject_id is None:
            continue

        pngs = sorted(pngs, key=get_slice_index)

        for path in pngs:
            slice_idx = get_slice_index(path)

            if slice_idx == 9999:
                slice_name = os.path.splitext(os.path.basename(path))[0]
            else:
                slice_name = f"slice_{slice_idx:02d}"

            rows.append(
                {
                    "subject_id": subject_id,
                    "sample_id": f"{subject_id}_{slice_name}",
                    "input_variant": slice_name,
                    "path_ch0": path,
                    "path_ch1": path,
                    "path_ch2": path,
                }
            )

    return pd.DataFrame(rows)


def build_patch_triplet_aug_samples(config):
    rows = []
    maps = {}

    for key in ["minus1", "center", "plus1"]:
        maps[key] = {}

        for path in sorted_pngs(config["paths"][key]):
            subject_id = extraer_subject_id(os.path.basename(path))

            if subject_id is None:
                continue

            variant = get_patch_variant(path)
            maps[key].setdefault((subject_id, variant), []).append(path)

    common_keys = sorted(set(maps["minus1"]) & set(maps["center"]) & set(maps["plus1"]))

    for subject_id, variant in common_keys:
        rows.append(
            {
                "subject_id": subject_id,
                "sample_id": f"{subject_id}_patch_{variant}",
                "input_variant": f"patch_{variant}",
                "path_ch0": choose_first(maps["minus1"][(subject_id, variant)]),
                "path_ch1": choose_first(maps["center"][(subject_id, variant)]),
                "path_ch2": choose_first(maps["plus1"][(subject_id, variant)]),
            }
        )

    return pd.DataFrame(rows)


def build_pve_channel_samples(config):
    root = config["paths"]["root"]
    pve_map = {}

    for path in sorted_pngs(root):
        subject_id = extraer_subject_id(os.path.basename(path))
        pve_idx = get_pve_index(path)

        if subject_id is None or pve_idx is None:
            continue

        pve_map.setdefault(subject_id, {}).setdefault(pve_idx, []).append(path)

    rows = []

    for subject_id in sorted(pve_map):
        if not all(idx in pve_map[subject_id] for idx in [0, 1, 2]):
            continue

        rows.append(
            {
                "subject_id": subject_id,
                "sample_id": f"{subject_id}_pve012",
                "input_variant": "pve_0_1_2",
                "path_ch0": choose_first(pve_map[subject_id][0]),
                "path_ch1": choose_first(pve_map[subject_id][1]),
                "path_ch2": choose_first(pve_map[subject_id][2]),
            }
        )

    return pd.DataFrame(rows)


def build_image_samples(config):
    kind = config["kind"]

    if kind == "single_slice":
        return build_single_slice_samples(config)

    if kind == "triplet":
        return build_triplet_samples(config)

    if kind == "slice_level":
        return build_slice_level_samples(config)

    if kind == "patch_triplet_aug":
        return build_patch_triplet_aug_samples(config)

    if kind == "pve_channels":
        return build_pve_channel_samples(config)

    raise ValueError(f"Input kind no reconocido: {kind}")


def attach_labels_and_split(samples_df, df_case, split_df):
    meta_cols = [
        "subject_id",
        "age",
        "age_group_name",
        "diagnosis_name",
        "label",
        "target_name",
    ]

    df = samples_df.merge(df_case[meta_cols], on="subject_id", how="inner")
    df = df.merge(split_df[["subject_id", "split"]], on="subject_id", how="inner")
    df = df.sort_values(["split", "subject_id", "sample_id"]).reset_index(drop=True)

    return df


def check_no_subject_leakage(df):
    split_by_subject = df.groupby("subject_id")["split"].nunique()
    leaked = split_by_subject[split_by_subject > 1]

    if len(leaked) > 0:
        raise ValueError(f"Data leakage detectado en sujetos: {leaked.index.tolist()[:10]}")


# =========================================================
# TF DATASET
# =========================================================
@tf.autograph.experimental.do_not_convert
def load_three_channel_png(path_ch0, path_ch1, path_ch2, label):
    img0 = tf.io.read_file(path_ch0)
    img0 = tf.image.decode_png(img0, channels=1)
    img0 = tf.image.resize(img0, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
    img0 = tf.cast(img0, tf.float32)

    img1 = tf.io.read_file(path_ch1)
    img1 = tf.image.decode_png(img1, channels=1)
    img1 = tf.image.resize(img1, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
    img1 = tf.cast(img1, tf.float32)

    img2 = tf.io.read_file(path_ch2)
    img2 = tf.image.decode_png(img2, channels=1)
    img2 = tf.image.resize(img2, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
    img2 = tf.cast(img2, tf.float32)

    img = tf.concat([img0, img1, img2], axis=-1)
    img = preprocess_input(img)

    return img, label


def make_dataset(df, shuffle=False):
    paths_ch0 = df["path_ch0"].values
    paths_ch1 = df["path_ch1"].values
    paths_ch2 = df["path_ch2"].values
    labels = df["label"].astype("float32").values

    ds = tf.data.Dataset.from_tensor_slices((paths_ch0, paths_ch1, paths_ch2, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(load_three_channel_png, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds


# =========================================================
# MODELO
# =========================================================
def build_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )

    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    return model


# =========================================================
# PREDICCIÓN Y AGREGACIÓN
# =========================================================
def predict_sample_level(model, df):
    ds = make_dataset(df, shuffle=False)
    y_prob = model.predict(ds, verbose=0).ravel()

    pred_df = df[
        [
            "subject_id",
            "sample_id",
            "input_variant",
            "split",
            "age",
            "age_group_name",
            "diagnosis_name",
            "label",
            "target_name",
        ]
    ].copy()

    pred_df["y_prob_class_1"] = y_prob
    pred_df["y_pred_sample"] = (pred_df["y_prob_class_1"] >= 0.5).astype(int)
    pred_df["pred_target_name_sample"] = pred_df["y_pred_sample"].map(
        {
            0: "young_CU",
            1: "old_AD",
        }
    )
    pred_df["correct_sample"] = (pred_df["label"] == pred_df["y_pred_sample"]).astype(int)

    return pred_df


def aggregate_to_subject_level(sample_pred_df):
    subject_pred_df = (
        sample_pred_df.groupby("subject_id", as_index=False)
        .agg(
            age=("age", "first"),
            age_group_name=("age_group_name", "first"),
            diagnosis_name=("diagnosis_name", "first"),
            label=("label", "first"),
            target_name=("target_name", "first"),
            split=("split", "first"),
            n_samples=("sample_id", "count"),
            y_prob_class_1=("y_prob_class_1", "mean"),
        )
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    subject_pred_df["y_pred"] = (subject_pred_df["y_prob_class_1"] >= 0.5).astype(int)
    subject_pred_df["pred_target_name"] = subject_pred_df["y_pred"].map(
        {
            0: "young_CU",
            1: "old_AD",
        }
    )
    subject_pred_df["correct"] = (
        subject_pred_df["label"] == subject_pred_df["y_pred"]
    ).astype(int)

    return subject_pred_df


# =========================================================
# MÉTRICAS + BOOTSTRAP
# =========================================================
def compute_basic_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        ),
        "auc": safe_auc(y_true, y_prob),
    }


def bootstrap_metric_distribution(y_true, y_pred, y_prob, n_bootstrap, random_state):
    n_subjects = len(y_true)
    rng = np.random.default_rng(random_state)

    bootstrap_rows = []

    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n_subjects), size=n_subjects, replace=True)

        metrics_i = compute_basic_metrics(
            y_true=y_true[idx],
            y_pred=y_pred[idx],
            y_prob=y_prob[idx],
        )

        bootstrap_rows.append(metrics_i)

    return pd.DataFrame(bootstrap_rows)


def evaluate_predictions(
    subject_pred_df,
    class_names,
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE,
):
    y_true = subject_pred_df["label"].values.astype(int)
    y_prob = subject_pred_df["y_prob_class_1"].values.astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    original_metrics = compute_basic_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    boot_df = bootstrap_metric_distribution(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    metrics = {
        "accuracy": original_metrics["accuracy"],
        "accuracy_mean": original_metrics["accuracy"],
        "accuracy_std": safe_nanstd(boot_df["accuracy"].values),
        "accuracy_bootstrap_mean": safe_nanmean(boot_df["accuracy"].values),
        "accuracy_ci95_low": safe_nanpercentile(boot_df["accuracy"].values, 2.5),
        "accuracy_ci95_high": safe_nanpercentile(boot_df["accuracy"].values, 97.5),

        "balanced_accuracy": original_metrics["balanced_accuracy"],
        "balanced_accuracy_mean": original_metrics["balanced_accuracy"],
        "balanced_accuracy_std": safe_nanstd(boot_df["balanced_accuracy"].values),
        "balanced_accuracy_bootstrap_mean": safe_nanmean(boot_df["balanced_accuracy"].values),
        "balanced_accuracy_ci95_low": safe_nanpercentile(boot_df["balanced_accuracy"].values, 2.5),
        "balanced_accuracy_ci95_high": safe_nanpercentile(boot_df["balanced_accuracy"].values, 97.5),

        "precision_macro": original_metrics["precision_macro"],
        "precision_macro_mean": original_metrics["precision_macro"],
        "precision_macro_std": safe_nanstd(boot_df["precision_macro"].values),
        "precision_macro_bootstrap_mean": safe_nanmean(boot_df["precision_macro"].values),
        "precision_macro_ci95_low": safe_nanpercentile(boot_df["precision_macro"].values, 2.5),
        "precision_macro_ci95_high": safe_nanpercentile(boot_df["precision_macro"].values, 97.5),

        "recall_macro": original_metrics["recall_macro"],
        "recall_macro_mean": original_metrics["recall_macro"],
        "recall_macro_std": safe_nanstd(boot_df["recall_macro"].values),
        "recall_macro_bootstrap_mean": safe_nanmean(boot_df["recall_macro"].values),
        "recall_macro_ci95_low": safe_nanpercentile(boot_df["recall_macro"].values, 2.5),
        "recall_macro_ci95_high": safe_nanpercentile(boot_df["recall_macro"].values, 97.5),

        "f1_macro": original_metrics["f1_macro"],
        "f1_macro_mean": original_metrics["f1_macro"],
        "f1_macro_std": safe_nanstd(boot_df["f1_macro"].values),
        "f1_macro_bootstrap_mean": safe_nanmean(boot_df["f1_macro"].values),
        "f1_macro_ci95_low": safe_nanpercentile(boot_df["f1_macro"].values, 2.5),
        "f1_macro_ci95_high": safe_nanpercentile(boot_df["f1_macro"].values, 97.5),

        "auc": original_metrics["auc"],
        "auc_mean": original_metrics["auc"],
        "auc_std": safe_nanstd(boot_df["auc"].values),
        "auc_bootstrap_mean": safe_nanmean(boot_df["auc"].values),
        "auc_ci95_low": safe_nanpercentile(boot_df["auc"].values, 2.5),
        "auc_ci95_high": safe_nanpercentile(boot_df["auc"].values, 97.5),

        "cm": cm,
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "bootstrap_df": boot_df,
    }

    return metrics


# =========================================================
# PLOTS
# =========================================================
def plot_learning_curves(history, mode_name, title, results_dir):
    epochs_range = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, history.history["loss"], label="train_loss")
    plt.plot(epochs_range, history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss | {title}")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"loss_curve_{mode_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, history.history["accuracy"], label="train_accuracy")
    plt.plot(epochs_range, history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy | {title}")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"accuracy_curve_{mode_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_single_confusion_matrix(
    metrics,
    class_names,
    split_name,
    mode_name,
    results_dir,
):
    fig, ax = plt.subplots(figsize=(4, 4))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["cm"],
        display_labels=class_names,
    )

    disp.plot(
        ax=ax,
        cmap="Purples",
        colorbar=False,
        values_format="d",
    )

    ax.set_title(split_name)

    for text in ax.texts:
        text.set_fontsize(14)

    plt.tight_layout()
    out_path = os.path.join(
        results_dir,
        f"confusion_matrix_{split_name.lower()}_{mode_name}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_train_test_conf_matrices(
    train_metrics,
    test_metrics,
    class_names,
    mode_name,
    title,
    results_dir,
):
    plot_single_confusion_matrix(
        metrics=train_metrics,
        class_names=class_names,
        split_name="Train",
        mode_name=mode_name,
        results_dir=results_dir,
    )

    plot_single_confusion_matrix(
        metrics=test_metrics,
        class_names=class_names,
        split_name="Test",
        mode_name=mode_name,
        results_dir=results_dir,
    )


def plot_roc_curve_test(test_metrics, mode_name, title, results_dir):
    y_true = test_metrics["y_true"]
    y_prob = test_metrics["y_prob"]

    if len(np.unique(y_true)) < 2:
        print("No se puede dibujar ROC porque test solo tiene una clase.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC | {title}")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"roc_curve_{mode_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================================
# IMPRESIÓN Y GUARDADO
# =========================================================
def print_subject_distribution(subject_df, split_name):
    print(f"\n{split_name}: {len(subject_df)} sujetos")

    print("Target:")
    print(
        subject_df["label"]
        .value_counts()
        .sort_index()
        .rename(index={0: "young_CU", 1: "old_AD"})
    )

    print("Diagnóstico:")
    print(subject_df["diagnosis_name"].value_counts())

    print("Age group:")
    print(subject_df["age_group_name"].value_counts())


def save_basic_tables(df_all, split_df, df_case, age_threshold, mode_name, results_dir):
    threshold_df = pd.DataFrame(
        [
            {
                "threshold_method": "mean",
                "age_threshold": age_threshold,
                "computed_from": "all_subjects_in_csv_with_valid_subject_age_dx",
                "n_subjects_for_threshold": len(df_all),
            }
        ]
    )

    threshold_df.to_csv(
        os.path.join(results_dir, f"age_threshold_{mode_name}.csv"),
        index=False,
    )

    split_df.to_csv(
        os.path.join(results_dir, f"subject_splits_{mode_name}.csv"),
        index=False,
    )

    df_case.to_csv(
        os.path.join(results_dir, f"subjects_case_{mode_name}.csv"),
        index=False,
    )

    label_map_df = pd.DataFrame(
        [
            {"label": 0, "class_name": "young_CU"},
            {"label": 1, "class_name": "old_AD"},
        ]
    )

    label_map_df.to_csv(
        os.path.join(results_dir, f"label_map_{mode_name}.csv"),
        index=False,
    )


def metrics_to_row(split_name, metrics, n_subjects):
    return {
        "split": split_name,
        "n_subjects": n_subjects,
        "n_bootstrap": N_BOOTSTRAP,

        "accuracy_mean": metrics["accuracy_mean"],
        "accuracy_std": metrics["accuracy_std"],
        "accuracy_bootstrap_mean": metrics["accuracy_bootstrap_mean"],
        "accuracy_ci95_low": metrics["accuracy_ci95_low"],
        "accuracy_ci95_high": metrics["accuracy_ci95_high"],

        "balanced_accuracy_mean": metrics["balanced_accuracy_mean"],
        "balanced_accuracy_std": metrics["balanced_accuracy_std"],
        "balanced_accuracy_bootstrap_mean": metrics["balanced_accuracy_bootstrap_mean"],
        "balanced_accuracy_ci95_low": metrics["balanced_accuracy_ci95_low"],
        "balanced_accuracy_ci95_high": metrics["balanced_accuracy_ci95_high"],

        "precision_macro_mean": metrics["precision_macro_mean"],
        "precision_macro_std": metrics["precision_macro_std"],
        "precision_macro_bootstrap_mean": metrics["precision_macro_bootstrap_mean"],
        "precision_macro_ci95_low": metrics["precision_macro_ci95_low"],
        "precision_macro_ci95_high": metrics["precision_macro_ci95_high"],

        "recall_macro_mean": metrics["recall_macro_mean"],
        "recall_macro_std": metrics["recall_macro_std"],
        "recall_macro_bootstrap_mean": metrics["recall_macro_bootstrap_mean"],
        "recall_macro_ci95_low": metrics["recall_macro_ci95_low"],
        "recall_macro_ci95_high": metrics["recall_macro_ci95_high"],

        "f1_macro_mean": metrics["f1_macro_mean"],
        "f1_macro_std": metrics["f1_macro_std"],
        "f1_macro_bootstrap_mean": metrics["f1_macro_bootstrap_mean"],
        "f1_macro_ci95_low": metrics["f1_macro_ci95_low"],
        "f1_macro_ci95_high": metrics["f1_macro_ci95_high"],

        "auc_mean": metrics["auc_mean"],
        "auc_std": metrics["auc_std"],
        "auc_bootstrap_mean": metrics["auc_bootstrap_mean"],
        "auc_ci95_low": metrics["auc_ci95_low"],
        "auc_ci95_high": metrics["auc_ci95_high"],
    }


def save_bootstrap_raw(metrics, split_name, mode_name, results_dir):
    out_path = os.path.join(
        results_dir,
        f"bootstrap_raw_{split_name}_{mode_name}.csv",
    )

    metrics["bootstrap_df"].to_csv(out_path, index=False)


def save_metrics_and_predictions(
    mode_name,
    title,
    results_dir,
    age_threshold,
    df_train,
    df_val,
    df_test,
    train_subject_pred_df,
    val_subject_pred_df,
    test_subject_pred_df,
    train_sample_pred_df,
    val_sample_pred_df,
    test_sample_pred_df,
    train_metrics,
    val_metrics,
    test_metrics,
    history,
):
    train_subject_pred_df.to_csv(
        os.path.join(results_dir, f"train_predictions_subject_level_{mode_name}.csv"),
        index=False,
    )

    val_subject_pred_df.to_csv(
        os.path.join(results_dir, f"val_predictions_subject_level_{mode_name}.csv"),
        index=False,
    )

    test_subject_pred_df.to_csv(
        os.path.join(results_dir, f"test_predictions_subject_level_{mode_name}.csv"),
        index=False,
    )

    train_sample_pred_df.to_csv(
        os.path.join(results_dir, f"train_predictions_sample_level_{mode_name}.csv"),
        index=False,
    )

    val_sample_pred_df.to_csv(
        os.path.join(results_dir, f"val_predictions_sample_level_{mode_name}.csv"),
        index=False,
    )

    test_sample_pred_df.to_csv(
        os.path.join(results_dir, f"test_predictions_sample_level_{mode_name}.csv"),
        index=False,
    )

    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = np.arange(1, len(history_df) + 1)
    history_df.to_csv(
        os.path.join(results_dir, f"training_history_{mode_name}.csv"),
        index=False,
    )

    report_train_df = pd.DataFrame(train_metrics["report"]).transpose()
    report_val_df = pd.DataFrame(val_metrics["report"]).transpose()
    report_test_df = pd.DataFrame(test_metrics["report"]).transpose()

    report_train_df.to_csv(
        os.path.join(results_dir, f"classification_report_train_{mode_name}.csv")
    )

    report_val_df.to_csv(
        os.path.join(results_dir, f"classification_report_val_{mode_name}.csv")
    )

    report_test_df.to_csv(
        os.path.join(results_dir, f"classification_report_test_{mode_name}.csv")
    )

    save_bootstrap_raw(train_metrics, "train", mode_name, results_dir)
    save_bootstrap_raw(val_metrics, "val", mode_name, results_dir)
    save_bootstrap_raw(test_metrics, "test", mode_name, results_dir)

    metrics_bootstrap_df = pd.DataFrame(
        [
            metrics_to_row(
                split_name="train",
                metrics=train_metrics,
                n_subjects=df_train["subject_id"].nunique(),
            ),
            metrics_to_row(
                split_name="val",
                metrics=val_metrics,
                n_subjects=df_val["subject_id"].nunique(),
            ),
            metrics_to_row(
                split_name="test",
                metrics=test_metrics,
                n_subjects=df_test["subject_id"].nunique(),
            ),
        ]
    )

    metrics_bootstrap_df.to_csv(
        os.path.join(results_dir, f"metrics_bootstrap_{mode_name}.csv"),
        index=False,
    )

    counts_train = train_subject_pred_df["label"].value_counts().sort_index().to_dict()
    counts_val = val_subject_pred_df["label"].value_counts().sort_index().to_dict()
    counts_test = test_subject_pred_df["label"].value_counts().sort_index().to_dict()

    total_subjects = pd.concat(
        [train_subject_pred_df, val_subject_pred_df, test_subject_pred_df],
        ignore_index=True,
    ).drop_duplicates("subject_id")

    counts_total = total_subjects["label"].value_counts().sort_index().to_dict()

    summary_df = pd.DataFrame(
        [
            {
                "mode": mode_name,
                "title": title,
                "case_id": "young_cu_vs_old_ad",
                "threshold_method": "mean",
                "age_threshold": age_threshold,
                "class_0": "young_CU",
                "class_1": "old_AD",
                "positive_class": "old_AD",
                "n_bootstrap": N_BOOTSTRAP,

                "n_train_subjects": train_subject_pred_df["subject_id"].nunique(),
                "n_val_subjects": val_subject_pred_df["subject_id"].nunique(),
                "n_test_subjects": test_subject_pred_df["subject_id"].nunique(),
                "n_total_subjects": total_subjects["subject_id"].nunique(),

                "n_train_samples": len(df_train),
                "n_val_samples": len(df_val),
                "n_test_samples": len(df_test),
                "n_total_samples": len(df_train) + len(df_val) + len(df_test),

                "n_total_class_0": counts_total.get(0, 0),
                "n_total_class_1": counts_total.get(1, 0),
                "n_train_class_0": counts_train.get(0, 0),
                "n_train_class_1": counts_train.get(1, 0),
                "n_val_class_0": counts_val.get(0, 0),
                "n_val_class_1": counts_val.get(1, 0),
                "n_test_class_0": counts_test.get(0, 0),
                "n_test_class_1": counts_test.get(1, 0),

                "test_accuracy": test_metrics["accuracy"],
                "test_accuracy_std": test_metrics["accuracy_std"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_balanced_accuracy_std": test_metrics["balanced_accuracy_std"],
                "test_precision_macro": test_metrics["precision_macro"],
                "test_precision_macro_std": test_metrics["precision_macro_std"],
                "test_recall_macro": test_metrics["recall_macro"],
                "test_recall_macro_std": test_metrics["recall_macro_std"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_f1_macro_std": test_metrics["f1_macro_std"],
                "test_auc": test_metrics["auc"],
                "test_auc_std": test_metrics["auc_std"],

                "val_accuracy": val_metrics["accuracy"],
                "val_accuracy_std": val_metrics["accuracy_std"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_balanced_accuracy_std": val_metrics["balanced_accuracy_std"],
                "val_precision_macro": val_metrics["precision_macro"],
                "val_precision_macro_std": val_metrics["precision_macro_std"],
                "val_recall_macro": val_metrics["recall_macro"],
                "val_recall_macro_std": val_metrics["recall_macro_std"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_f1_macro_std": val_metrics["f1_macro_std"],
                "val_auc": val_metrics["auc"],
                "val_auc_std": val_metrics["auc_std"],

                "train_accuracy": train_metrics["accuracy"],
                "train_accuracy_std": train_metrics["accuracy_std"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_balanced_accuracy_std": train_metrics["balanced_accuracy_std"],
                "train_precision_macro": train_metrics["precision_macro"],
                "train_precision_macro_std": train_metrics["precision_macro_std"],
                "train_recall_macro": train_metrics["recall_macro"],
                "train_recall_macro_std": train_metrics["recall_macro_std"],
                "train_f1_macro": train_metrics["f1_macro"],
                "train_f1_macro_std": train_metrics["f1_macro_std"],
                "train_auc": train_metrics["auc"],
                "train_auc_std": train_metrics["auc_std"],
            }
        ]
    )

    summary_df.to_csv(
        os.path.join(results_dir, f"summary_results_{mode_name}.csv"),
        index=False,
    )

    print("\nMÉTRICAS TEST, subject-level")
    print("Formato: metric ± bootstrap std")
    print(f"Accuracy:        {test_metrics['accuracy_mean']:.4f} ± {test_metrics['accuracy_std']:.4f}")
    print(f"Balanced acc:    {test_metrics['balanced_accuracy_mean']:.4f} ± {test_metrics['balanced_accuracy_std']:.4f}")
    print(f"Precision macro: {test_metrics['precision_macro_mean']:.4f} ± {test_metrics['precision_macro_std']:.4f}")
    print(f"Recall macro:    {test_metrics['recall_macro_mean']:.4f} ± {test_metrics['recall_macro_std']:.4f}")
    print(f"F1 macro:        {test_metrics['f1_macro_mean']:.4f} ± {test_metrics['f1_macro_std']:.4f}")
    print(f"AUC:             {test_metrics['auc_mean']:.4f} ± {test_metrics['auc_std']:.4f}")

    print("\nClassification report test, subject-level:")
    print(report_test_df)

    print("\nConfusion matrix test, subject-level:")
    print(test_metrics["cm"])


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(RANDOM_STATE)

    if ACTIVE_INPUT_MODE not in INPUT_CONFIGS:
        raise ValueError(f"ACTIVE_INPUT_MODE no reconocido: {ACTIVE_INPUT_MODE}")

    config = INPUT_CONFIGS[ACTIVE_INPUT_MODE]
    title = config["title"]
    mode_name = ACTIVE_INPUT_MODE
    results_dir = os.path.join(RESULTS_ROOT, mode_name)
    os.makedirs(results_dir, exist_ok=True)

    class_names = ["young_CU", "old_AD"]

    print("=" * 100)
    print(f"EXPERIMENTO: {title}")
    print("=" * 100)
    print(f"Resultados en: {results_dir}")
    print(f"Bootstrap: {N_BOOTSTRAP} resamples por split")

    df_metadata = read_subject_metadata()
    age_threshold = float(df_metadata["age"].mean())
    df_case = prepare_young_cu_vs_old_ad_from_csv(df_metadata, age_threshold)
    split_df = make_fixed_subject_splits(df_case)

    print(f"\nThreshold mean(AGE) calculado con todos los sujetos del CSV: {age_threshold:.4f}")
    print("Definición edad: young = AGE <= threshold | old = AGE > threshold")
    print("Label 0 = young_CU | Label 1 = old_AD")

    for split_name in ["train", "val", "test"]:
        split_subjects = split_df[split_df["split"] == split_name]
        print_subject_distribution(split_subjects, split_name.upper())

    save_basic_tables(
        df_all=df_metadata,
        split_df=split_df,
        df_case=df_case,
        age_threshold=age_threshold,
        mode_name=mode_name,
        results_dir=results_dir,
    )

    samples_df = build_image_samples(config)
    samples_df = attach_labels_and_split(samples_df, df_case, split_df)
    check_no_subject_leakage(samples_df)

    if samples_df.empty:
        raise ValueError(
            "No hay muestras de imagen para este modo después de cruzar con el CSV y el caso."
        )

    samples_df.to_csv(
        os.path.join(results_dir, f"image_samples_{mode_name}.csv"),
        index=False,
    )

    df_train = samples_df[samples_df["split"] == "train"].reset_index(drop=True)
    df_val = samples_df[samples_df["split"] == "val"].reset_index(drop=True)
    df_test = samples_df[samples_df["split"] == "test"].reset_index(drop=True)

    print("\nMuestras de imagen:")
    print(f"Train: {len(df_train)} muestras | {df_train['subject_id'].nunique()} sujetos")
    print(f"Val:   {len(df_val)} muestras | {df_val['subject_id'].nunique()} sujetos")
    print(f"Test:  {len(df_test)} muestras | {df_test['subject_id'].nunique()} sujetos")

    train_ids = set(df_train["subject_id"])
    val_ids = set(df_val["subject_id"])
    test_ids = set(df_test["subject_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    ds_train = make_dataset(df_train, shuffle=True)
    ds_val = make_dataset(df_val, shuffle=False)

    tf.keras.backend.clear_session()
    gc.collect()

    model = build_model()

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        verbose=1,
    )

    train_sample_pred_df = predict_sample_level(model, df_train)
    val_sample_pred_df = predict_sample_level(model, df_val)
    test_sample_pred_df = predict_sample_level(model, df_test)

    train_subject_pred_df = aggregate_to_subject_level(train_sample_pred_df)
    val_subject_pred_df = aggregate_to_subject_level(val_sample_pred_df)
    test_subject_pred_df = aggregate_to_subject_level(test_sample_pred_df)

    train_metrics = evaluate_predictions(
        train_subject_pred_df,
        class_names,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE,
    )

    val_metrics = evaluate_predictions(
        val_subject_pred_df,
        class_names,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE,
    )

    test_metrics = evaluate_predictions(
        test_subject_pred_df,
        class_names,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE,
    )

    plot_learning_curves(
        history=history,
        mode_name=mode_name,
        title=title,
        results_dir=results_dir,
    )

    plot_train_test_conf_matrices(
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        class_names=class_names,
        mode_name=mode_name,
        title=title,
        results_dir=results_dir,
    )

    plot_roc_curve_test(
        test_metrics=test_metrics,
        mode_name=mode_name,
        title=title,
        results_dir=results_dir,
    )

    save_metrics_and_predictions(
        mode_name=mode_name,
        title=title,
        results_dir=results_dir,
        age_threshold=age_threshold,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        train_subject_pred_df=train_subject_pred_df,
        val_subject_pred_df=val_subject_pred_df,
        test_subject_pred_df=test_subject_pred_df,
        train_sample_pred_df=train_sample_pred_df,
        val_sample_pred_df=val_sample_pred_df,
        test_sample_pred_df=test_sample_pred_df,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        history=history,
    )

    model.save(os.path.join(results_dir, f"model_{mode_name}.keras"))

    del model, ds_train, ds_val, history
    tf.keras.backend.clear_session()
    gc.collect()

    print("\nExperimento terminado.")
    print(f"Resultados guardados en: {results_dir}")


if __name__ == "__main__":
    main()