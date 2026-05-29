#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:01:34 2026

@author: irene
"""

"""
This script trains a ResNet50 model to classify A4/LEARN subjects using three central
coronal MRI slices as input channels. It creates subject-level train, validation and
test splits, trains the model, evaluates its performance, and estimates the standard
deviation of the metrics using bootstrap resampling of the subjects.
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
CSV_PATH = "/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/subjects_all.csv"
IMAGES_ROOT = "/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/coronal_resized_final"
RESULTS_ROOT = "/pool/home/AD_Multimodal/Estudio_A4/folder_irene/structural/results_sex"

SUBJECT_COL = "name_subject"
GROUP_COL = "group"
AGE_COL = "AGEYR"
SEX_COL = "SEX"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 1e-6
RANDOM_STATE = 42

TEST_SIZE = 0.20
VAL_SIZE_FROM_TRAIN = 0.20
MIN_SUBJECTS_PER_CLASS_FOR_SPLIT = 5
N_CENTRAL_SLICES = 3

# Bootstrap
N_BOOTSTRAP = 2000

plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


# =========================================================
# SELECCIONA SOLO UNA TAREA
# Deja descomentada solo una de estas 3 líneas.
# =========================================================
# ACTIVE_TASK = "a4_vs_learn"
# ACTIVE_TASK = "young_vs_old"
ACTIVE_TASK = "female_vs_male"


TASK_CONFIGS = {
    "a4_vs_learn": {
        "title": "A4 vs LEARN",
        "case_id": "a4_vs_learn",
        "class_names": ["a4", "learn"],
        "positive_class": "learn",
    },
    "young_vs_old": {
        "title": "Young vs Old",
        "case_id": "young_vs_old",
        "class_names": ["young", "old"],
        "positive_class": "old",
    },
    "female_vs_male": {
        "title": "Female vs Male",
        "case_id": "female_vs_male",
        "class_names": ["female", "male"],
        "positive_class": "male",
    },
}


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def normalizar_subject_id(x):
    return str(x).strip()


def normalizar_group(x):
    return str(x).strip().lower()


def sorted_pngs(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".png")
    )


def get_slice_index(path):
    name = os.path.basename(path).lower()

    match = re.search(r"slice[_-]?(\d+)", name)
    if match:
        return int(match.group(1))

    numbers = re.findall(r"\d+", name)
    if numbers:
        return int(numbers[-1])

    return 9999


def select_central_slices(pngs, n_central_slices=3):
    pngs = sorted(pngs, key=get_slice_index)

    if len(pngs) < n_central_slices:
        return []

    center = len(pngs) // 2
    start = center - (n_central_slices // 2)
    end = start + n_central_slices

    return pngs[start:end]


def slice_name_from_path(path):
    slice_idx = get_slice_index(path)

    if slice_idx == 9999:
        return os.path.splitext(os.path.basename(path))[0]

    return f"slice_{slice_idx:02d}"


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


def safe_nanpercentile(values, percentile):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan

    return float(np.percentile(values, percentile))


# =========================================================
# METADATA Y LABELS
# =========================================================
def read_subject_metadata():
    df_csv = pd.read_csv(CSV_PATH, low_memory=False)

    required_cols = [SUBJECT_COL, GROUP_COL, AGE_COL, SEX_COL]
    missing_cols = [col for col in required_cols if col not in df_csv.columns]

    if missing_cols:
        raise ValueError(f"Faltan estas columnas en el CSV: {missing_cols}")

    df = df_csv[required_cols].copy()

    df["subject_id"] = df[SUBJECT_COL].apply(normalizar_subject_id)
    df["group"] = df[GROUP_COL].apply(normalizar_group)
    df["AGEYR"] = pd.to_numeric(df[AGE_COL], errors="coerce")
    df["SEX"] = pd.to_numeric(df[SEX_COL], errors="coerce")

    df = df.dropna(subset=["subject_id", "group", "AGEYR", "SEX"])
    df = df.drop_duplicates(subset=["subject_id"])
    df["SEX"] = df["SEX"].astype(int)

    df = df[["subject_id", "group", "AGEYR", "SEX"]]
    df = df.sort_values("subject_id").reset_index(drop=True)

    return df


def prepare_a4_vs_learn_from_csv(df_metadata):
    df = df_metadata.copy()

    df["label"] = df["group"].map({"a4": 0, "learn": 1})
    df["target_name"] = df["label"].map({0: "a4", 1: "learn"})

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df.sort_values("subject_id").reset_index(drop=True)

    return df


def prepare_young_vs_old_from_csv(df_metadata):
    df = df_metadata.copy()

    age_threshold_mean = df["AGEYR"].mean()

    df["label"] = np.where(df["AGEYR"] >= age_threshold_mean, 1, 0)
    df["target_name"] = df["label"].map({0: "young", 1: "old"})
    df["age_threshold_mean"] = age_threshold_mean

    df = df.sort_values("subject_id").reset_index(drop=True)

    return df


def prepare_female_vs_male_from_csv(df_metadata):
    df = df_metadata.copy()

    df["label"] = df["SEX"].map({1: 0, 2: 1})
    df["target_name"] = df["label"].map({0: "female", 1: "male"})

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df.sort_values("subject_id").reset_index(drop=True)

    return df


def prepare_task_from_csv(df_metadata, task_name):
    if task_name == "a4_vs_learn":
        return prepare_a4_vs_learn_from_csv(df_metadata)

    if task_name == "young_vs_old":
        return prepare_young_vs_old_from_csv(df_metadata)

    if task_name == "female_vs_male":
        return prepare_female_vs_male_from_csv(df_metadata)

    raise ValueError(f"ACTIVE_TASK no reconocido: {task_name}")


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
    meta_cols = [
        "subject_id",
        "group",
        "AGEYR",
        "SEX",
        "label",
        "target_name",
    ]

    if "age_threshold_mean" in df_case.columns:
        meta_cols.append("age_threshold_mean")

    subjects_df = df_case[meta_cols].drop_duplicates("subject_id")
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
# CREACIÓN DE DATAFRAME DE IMÁGENES
# =========================================================
def build_subject_level_samples_from_root(root):
    rows = []

    for subject_id in sorted(os.listdir(root)):
        subject_folder = os.path.join(root, subject_id)

        if not os.path.isdir(subject_folder):
            continue

        pngs = sorted_pngs(subject_folder)
        central_pngs = select_central_slices(
            pngs=pngs,
            n_central_slices=N_CENTRAL_SLICES,
        )

        if len(central_pngs) != N_CENTRAL_SLICES:
            print(f"Se omite {subject_id}: no tiene {N_CENTRAL_SLICES} slices centrales.")
            continue

        path_ch0, path_ch1, path_ch2 = central_pngs

        rows.append(
            {
                "subject_id": subject_id,
                "sample_id": subject_id,
                "input_variant": "3central_slices_as_channels",
                "slice_ch0": slice_name_from_path(path_ch0),
                "slice_ch1": slice_name_from_path(path_ch1),
                "slice_ch2": slice_name_from_path(path_ch2),
                "path_ch0": path_ch0,
                "path_ch1": path_ch1,
                "path_ch2": path_ch2,
            }
        )

    samples_df = pd.DataFrame(rows)
    samples_df = samples_df.sort_values("subject_id").reset_index(drop=True)

    return samples_df


def attach_labels_and_split(samples_df, df_case, split_df):
    meta_cols = [
        "subject_id",
        "group",
        "AGEYR",
        "SEX",
        "label",
        "target_name",
    ]

    if "age_threshold_mean" in df_case.columns:
        meta_cols.append("age_threshold_mean")

    df = samples_df.merge(df_case[meta_cols], on="subject_id", how="inner")
    df = df.merge(split_df[["subject_id", "split"]], on="subject_id", how="inner")

    df = df.sort_values(["split", "subject_id"]).reset_index(drop=True)

    return df


def check_no_subject_leakage(df):
    split_by_subject = df.groupby("subject_id")["split"].nunique()
    leaked = split_by_subject[split_by_subject > 1]

    if len(leaked) > 0:
        raise ValueError(
            f"Data leakage detectado en sujetos: {leaked.index.tolist()[:10]}"
        )


# =========================================================
# TF DATASET
# =========================================================
@tf.autograph.experimental.do_not_convert
def load_three_slices_as_channels(path_ch0, path_ch1, path_ch2, label):
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

    ds = tf.data.Dataset.from_tensor_slices(
        (paths_ch0, paths_ch1, paths_ch2, labels)
    )

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(load_three_slices_as_channels, num_parallel_calls=tf.data.AUTOTUNE)
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
# PREDICCIÓN Y MÉTRICAS
# =========================================================
def predict_subject_level(model, df, class_names):
    ds = make_dataset(df, shuffle=False)
    y_prob = model.predict(ds, verbose=0).ravel()

    pred_cols = [
        "subject_id",
        "sample_id",
        "input_variant",
        "slice_ch0",
        "slice_ch1",
        "slice_ch2",
        "path_ch0",
        "path_ch1",
        "path_ch2",
        "split",
        "group",
        "AGEYR",
        "SEX",
        "label",
        "target_name",
    ]

    if "age_threshold_mean" in df.columns:
        pred_cols.append("age_threshold_mean")

    pred_df = df[pred_cols].copy()

    pred_df["y_prob_class_1"] = y_prob
    pred_df["y_pred"] = (pred_df["y_prob_class_1"] >= 0.5).astype(int)

    pred_df["pred_target_name"] = pred_df["y_pred"].map(
        {
            0: class_names[0],
            1: class_names[1],
        }
    )

    pred_df["correct"] = (pred_df["label"] == pred_df["y_pred"]).astype(int)

    return pred_df


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

    rows = []

    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n_subjects), size=n_subjects, replace=True)

        metrics_i = compute_basic_metrics(
            y_true=y_true[idx],
            y_pred=y_pred[idx],
            y_prob=y_prob[idx],
        )

        rows.append(metrics_i)

    return pd.DataFrame(rows)


def evaluate_predictions(subject_pred_df, class_names, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE):
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
        "accuracy_bootstrap_mean": float(np.nanmean(boot_df["accuracy"].values)),
        "accuracy_ci95_low": safe_nanpercentile(boot_df["accuracy"].values, 2.5),
        "accuracy_ci95_high": safe_nanpercentile(boot_df["accuracy"].values, 97.5),

        "balanced_accuracy": original_metrics["balanced_accuracy"],
        "balanced_accuracy_mean": original_metrics["balanced_accuracy"],
        "balanced_accuracy_std": safe_nanstd(boot_df["balanced_accuracy"].values),
        "balanced_accuracy_bootstrap_mean": float(np.nanmean(boot_df["balanced_accuracy"].values)),
        "balanced_accuracy_ci95_low": safe_nanpercentile(boot_df["balanced_accuracy"].values, 2.5),
        "balanced_accuracy_ci95_high": safe_nanpercentile(boot_df["balanced_accuracy"].values, 97.5),

        "precision_macro": original_metrics["precision_macro"],
        "precision_macro_mean": original_metrics["precision_macro"],
        "precision_macro_std": safe_nanstd(boot_df["precision_macro"].values),
        "precision_macro_bootstrap_mean": float(np.nanmean(boot_df["precision_macro"].values)),
        "precision_macro_ci95_low": safe_nanpercentile(boot_df["precision_macro"].values, 2.5),
        "precision_macro_ci95_high": safe_nanpercentile(boot_df["precision_macro"].values, 97.5),

        "recall_macro": original_metrics["recall_macro"],
        "recall_macro_mean": original_metrics["recall_macro"],
        "recall_macro_std": safe_nanstd(boot_df["recall_macro"].values),
        "recall_macro_bootstrap_mean": float(np.nanmean(boot_df["recall_macro"].values)),
        "recall_macro_ci95_low": safe_nanpercentile(boot_df["recall_macro"].values, 2.5),
        "recall_macro_ci95_high": safe_nanpercentile(boot_df["recall_macro"].values, 97.5),

        "f1_macro": original_metrics["f1_macro"],
        "f1_macro_mean": original_metrics["f1_macro"],
        "f1_macro_std": safe_nanstd(boot_df["f1_macro"].values),
        "f1_macro_bootstrap_mean": float(np.nanmean(boot_df["f1_macro"].values)),
        "f1_macro_ci95_low": safe_nanpercentile(boot_df["f1_macro"].values, 2.5),
        "f1_macro_ci95_high": safe_nanpercentile(boot_df["f1_macro"].values, 97.5),

        "auc": original_metrics["auc"],
        "auc_mean": original_metrics["auc"],
        "auc_std": safe_nanstd(boot_df["auc"].values),
        "auc_bootstrap_mean": float(np.nanmean(boot_df["auc"].values)),
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
def plot_learning_curves(history, experiment_name, title, results_dir):
    epochs_range = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, history.history["loss"], label="train_loss")
    plt.plot(epochs_range, history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"loss_curve_{experiment_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, history.history["accuracy"], label="train_accuracy")
    plt.plot(epochs_range, history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"accuracy_curve_{experiment_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_single_confusion_matrix(
    metrics,
    class_names,
    split_name,
    experiment_name,
    results_dir,
):
    labels_display = class_names

    fig, ax = plt.subplots(figsize=(4, 4))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["cm"],
        display_labels=labels_display,
    )

    disp.plot(
        ax=ax,
        cmap="Purples",
        colorbar=False,
        values_format="d",
    )

    ax.set_yticklabels(
        labels_display,
        rotation=90,
        va="center",
        ha="center",
    )
    ax.tick_params(axis="y", pad=8)

    for text in ax.texts:
        text.set_fontsize(14)

    plt.tight_layout()

    out_path = os.path.join(
        results_dir,
        f"confusion_matrix_{split_name.lower()}_{experiment_name}.png",
    )

    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_train_test_conf_matrices(
    train_metrics,
    test_metrics,
    class_names,
    experiment_name,
    results_dir,
    title=None,
):
    plot_single_confusion_matrix(
        metrics=train_metrics,
        class_names=class_names,
        split_name="Train",
        experiment_name=experiment_name,
        results_dir=results_dir,
    )

    plot_single_confusion_matrix(
        metrics=test_metrics,
        class_names=class_names,
        split_name="Test",
        experiment_name=experiment_name,
        results_dir=results_dir,
    )


def plot_roc_curve_test(test_metrics, experiment_name, title, results_dir):
    y_true = test_metrics["y_true"]
    y_prob = test_metrics["y_prob"]

    if len(np.unique(y_true)) < 2:
        print("No se puede dibujar ROC porque test solo tiene una clase.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"roc_curve_{experiment_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================================
# IMPRESIÓN Y GUARDADO
# =========================================================
def print_subject_distribution(subject_df, split_name, class_names):
    print(f"\n{split_name}: {len(subject_df)} sujetos")

    print("Target:")
    print(
        subject_df["label"]
        .value_counts()
        .sort_index()
        .rename(index={0: class_names[0], 1: class_names[1]})
    )

    print("group:")
    print(subject_df["group"].value_counts())

    print("SEX:")
    print(subject_df["SEX"].value_counts(dropna=False).rename(index={1: "female", 2: "male"}))

    print("AGEYR:")
    print(subject_df["AGEYR"].describe())


def save_basic_tables(
    df_metadata,
    df_case,
    split_df,
    samples_df,
    experiment_name,
    task_name,
    task_config,
    results_dir,
):
    df_metadata.to_csv(
        os.path.join(results_dir, f"metadata_all_subjects_{experiment_name}.csv"),
        index=False,
    )

    df_case.to_csv(
        os.path.join(results_dir, f"subjects_case_{experiment_name}.csv"),
        index=False,
    )

    split_df.to_csv(
        os.path.join(results_dir, f"subject_splits_{experiment_name}.csv"),
        index=False,
    )

    samples_df.to_csv(
        os.path.join(results_dir, f"model_input_subjects_{experiment_name}.csv"),
        index=False,
    )

    class_names = task_config["class_names"]

    label_map_df = pd.DataFrame(
        [
            {"label": 0, "class_name": class_names[0]},
            {"label": 1, "class_name": class_names[1]},
        ]
    )

    label_map_df.to_csv(
        os.path.join(results_dir, f"label_map_{experiment_name}.csv"),
        index=False,
    )

    age_threshold_mean = np.nan
    if "age_threshold_mean" in df_case.columns:
        age_threshold_mean = df_case["age_threshold_mean"].iloc[0]

    task_info_df = pd.DataFrame(
        [
            {
                "experiment_name": experiment_name,
                "task_name": task_name,
                "task_title": task_config["title"],
                "case_id": task_config["case_id"],
                "input_title": "3 central coronal slices as 3 channels",
                "n_central_slices": N_CENTRAL_SLICES,
                "one_sample_per_subject": True,
                "class_0": class_names[0],
                "class_1": class_names[1],
                "positive_class": task_config["positive_class"],
                "age_threshold_mean": age_threshold_mean,
                "n_bootstrap": N_BOOTSTRAP,
                "csv_path": CSV_PATH,
                "images_root": IMAGES_ROOT,
                "results_root": RESULTS_ROOT,
            }
        ]
    )

    task_info_df.to_csv(
        os.path.join(results_dir, f"task_info_{experiment_name}.csv"),
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


def save_bootstrap_raw(metrics, split_name, experiment_name, results_dir):
    boot_df = metrics["bootstrap_df"]

    out_path = os.path.join(
        results_dir,
        f"bootstrap_raw_{split_name}_{experiment_name}.csv",
    )

    boot_df.to_csv(out_path, index=False)


def save_metrics_and_predictions(
    experiment_name,
    task_config,
    results_dir,
    df_train,
    df_val,
    df_test,
    train_subject_pred_df,
    val_subject_pred_df,
    test_subject_pred_df,
    train_metrics,
    val_metrics,
    test_metrics,
    history,
):
    class_names = task_config["class_names"]

    train_subject_pred_df.to_csv(
        os.path.join(results_dir, f"train_predictions_subject_level_{experiment_name}.csv"),
        index=False,
    )

    val_subject_pred_df.to_csv(
        os.path.join(results_dir, f"val_predictions_subject_level_{experiment_name}.csv"),
        index=False,
    )

    test_subject_pred_df.to_csv(
        os.path.join(results_dir, f"test_predictions_subject_level_{experiment_name}.csv"),
        index=False,
    )

    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = np.arange(1, len(history_df) + 1)

    history_df.to_csv(
        os.path.join(results_dir, f"training_history_{experiment_name}.csv"),
        index=False,
    )

    report_train_df = pd.DataFrame(train_metrics["report"]).transpose()
    report_val_df = pd.DataFrame(val_metrics["report"]).transpose()
    report_test_df = pd.DataFrame(test_metrics["report"]).transpose()

    report_train_df.to_csv(
        os.path.join(results_dir, f"classification_report_train_{experiment_name}.csv")
    )

    report_val_df.to_csv(
        os.path.join(results_dir, f"classification_report_val_{experiment_name}.csv")
    )

    report_test_df.to_csv(
        os.path.join(results_dir, f"classification_report_test_{experiment_name}.csv")
    )

    save_bootstrap_raw(train_metrics, "train", experiment_name, results_dir)
    save_bootstrap_raw(val_metrics, "val", experiment_name, results_dir)
    save_bootstrap_raw(test_metrics, "test", experiment_name, results_dir)

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
        os.path.join(results_dir, f"metrics_bootstrap_{experiment_name}.csv"),
        index=False,
    )

    counts_train = df_train["label"].value_counts().sort_index().to_dict()
    counts_val = df_val["label"].value_counts().sort_index().to_dict()
    counts_test = df_test["label"].value_counts().sort_index().to_dict()

    total_subjects = pd.concat(
        [df_train, df_val, df_test],
        ignore_index=True,
    ).drop_duplicates("subject_id")

    counts_total = total_subjects["label"].value_counts().sort_index().to_dict()

    summary_df = pd.DataFrame(
        [
            {
                "experiment_name": experiment_name,
                "task_title": task_config["title"],
                "case_id": task_config["case_id"],
                "input_title": "3 central coronal slices as 3 channels",
                "one_sample_per_subject": True,
                "class_0": class_names[0],
                "class_1": class_names[1],
                "positive_class": task_config["positive_class"],
                "n_central_slices": N_CENTRAL_SLICES,
                "n_bootstrap": N_BOOTSTRAP,

                "n_train_subjects": df_train["subject_id"].nunique(),
                "n_val_subjects": df_val["subject_id"].nunique(),
                "n_test_subjects": df_test["subject_id"].nunique(),
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
        os.path.join(results_dir, f"summary_results_{experiment_name}.csv"),
        index=False,
    )

    print("\nMÉTRICAS TEST, subject-level")
    print("Formato: metric ± bootstrap std")
    print(
        f"Accuracy:        {test_metrics['accuracy_mean']:.4f} ± "
        f"{test_metrics['accuracy_std']:.4f}"
    )
    print(
        f"Balanced acc:    {test_metrics['balanced_accuracy_mean']:.4f} ± "
        f"{test_metrics['balanced_accuracy_std']:.4f}"
    )
    print(
        f"Precision macro: {test_metrics['precision_macro_mean']:.4f} ± "
        f"{test_metrics['precision_macro_std']:.4f}"
    )
    print(
        f"Recall macro:    {test_metrics['recall_macro_mean']:.4f} ± "
        f"{test_metrics['recall_macro_std']:.4f}"
    )
    print(
        f"F1 macro:        {test_metrics['f1_macro_mean']:.4f} ± "
        f"{test_metrics['f1_macro_std']:.4f}"
    )
    print(
        f"AUC:             {test_metrics['auc_mean']:.4f} ± "
        f"{test_metrics['auc_std']:.4f}"
    )

    print("\nClassification report test, subject-level:")
    print(report_test_df)

    print("\nConfusion matrix test, subject-level:")
    print(test_metrics["cm"])


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(RANDOM_STATE)

    if ACTIVE_TASK not in TASK_CONFIGS:
        raise ValueError(f"ACTIVE_TASK no reconocido: {ACTIVE_TASK}")

    task_config = TASK_CONFIGS[ACTIVE_TASK]
    title = f"{task_config['title']} | 3 central coronal slices as channels"
    class_names = task_config["class_names"]
    experiment_name = ACTIVE_TASK

    results_dir = os.path.join(RESULTS_ROOT, experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 100)
    print(f"EXPERIMENTO: {title}")
    print("=" * 100)
    print(f"Resultados en: {results_dir}")
    print(f"Tarea: {ACTIVE_TASK}")
    print(f"Clase 0 = {class_names[0]}")
    print(f"Clase 1 = {class_names[1]}")
    print(f"Clase positiva = {task_config['positive_class']}")
    print("Input: 1 sujeto = 1 muestra = 3 slices centrales como 3 canales")
    print(f"Bootstrap: {N_BOOTSTRAP} resamples por split")

    df_metadata = read_subject_metadata()
    df_case = prepare_task_from_csv(df_metadata, ACTIVE_TASK)

    samples_all_df = build_subject_level_samples_from_root(IMAGES_ROOT)

    subjects_with_images = set(samples_all_df["subject_id"].unique())
    df_case = df_case[df_case["subject_id"].isin(subjects_with_images)].copy()
    df_case = df_case.sort_values("subject_id").reset_index(drop=True)

    if df_case.empty:
        raise ValueError(
            "No hay sujetos después de cruzar el CSV con las carpetas de imágenes."
        )

    split_df = make_fixed_subject_splits(df_case)

    for split_name in ["train", "val", "test"]:
        split_subjects = split_df[split_df["split"] == split_name]

        print_subject_distribution(
            split_subjects,
            split_name.upper(),
            class_names,
        )

    samples_df = attach_labels_and_split(samples_all_df, df_case, split_df)
    check_no_subject_leakage(samples_df)

    if samples_df.empty:
        raise ValueError(
            "No hay muestras de imagen después de cruzar imágenes, labels y split."
        )

    save_basic_tables(
        df_metadata=df_metadata,
        df_case=df_case,
        split_df=split_df,
        samples_df=samples_df,
        experiment_name=experiment_name,
        task_name=ACTIVE_TASK,
        task_config=task_config,
        results_dir=results_dir,
    )

    df_train = samples_df[samples_df["split"] == "train"].reset_index(drop=True)
    df_val = samples_df[samples_df["split"] == "val"].reset_index(drop=True)
    df_test = samples_df[samples_df["split"] == "test"].reset_index(drop=True)

    print("\nMuestras de entrada al modelo:")
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

    train_subject_pred_df = predict_subject_level(
        model,
        df_train,
        class_names,
    )

    val_subject_pred_df = predict_subject_level(
        model,
        df_val,
        class_names,
    )

    test_subject_pred_df = predict_subject_level(
        model,
        df_test,
        class_names,
    )

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
        experiment_name=experiment_name,
        title=title,
        results_dir=results_dir,
    )

    plot_train_test_conf_matrices(
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        class_names=class_names,
        experiment_name=experiment_name,
        title=title,
        results_dir=results_dir,
    )

    plot_roc_curve_test(
        test_metrics=test_metrics,
        experiment_name=experiment_name,
        title=title,
        results_dir=results_dir,
    )

    save_metrics_and_predictions(
        experiment_name=experiment_name,
        task_config=task_config,
        results_dir=results_dir,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        train_subject_pred_df=train_subject_pred_df,
        val_subject_pred_df=val_subject_pred_df,
        test_subject_pred_df=test_subject_pred_df,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        history=history,
    )

    model.save(os.path.join(results_dir, f"model_{experiment_name}.keras"))

    del model, ds_train, ds_val, history
    tf.keras.backend.clear_session()
    gc.collect()

    print("\nExperimento terminado.")
    print(f"Resultados guardados en: {results_dir}")


if __name__ == "__main__":
    main()