#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs hyperparameter tuning for a ResNet50 model using the
coronal three-slice input configuration. It trains all hyperparameter
combinations with the same number of epochs, evaluates model performance on
the validation set, and selects the final configuration based on validation
AUC and validation loss stability. The test set is only used once, for the
final evaluation of the selected model.
"""

# =========================================================
# IMPORTS
# =========================================================
import os
import gc
import traceback

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, roc_auc_score

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

# Importa tu script original
import ResNet50_different_inputs as base


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
ACTIVE_INPUT_MODE = "coronal_3slices"

TUNING_RESULTS_ROOT = os.path.join(
    base.RESULTS_ROOT,
    "grid_tuning_coronal_3slices_stable"
)

os.makedirs(TUNING_RESULTS_ROOT, exist_ok=True)

CLASS_NAMES = ["young_CU", "old_AD"]

# Todas las combinaciones usan siempre el mismo número de epochs.
EPOCHS = 60

# No se evalúa test durante el tuning.
EVALUATE_TEST_FOR_TUNING = False

# Referencia visual anterior para 3 coronal slices, solo para dejarla anotada en el CSV final.
REFERENCE_TEST_ACCURACY = 0.76
REFERENCE_TEST_AUC = 0.824


# =========================================================
# HIPERPARÁMETROS A PROBAR
# =========================================================
# Se eliminan learning rates altos que pueden producir oscilaciones o subida de val_loss.
LEARNING_RATES_TO_TEST = [
    1e-7,
    3e-7,
    1e-6,
]

# Se elimina dropout 0.5 porque fue parte de la configuración inestable.
DROPOUTS_TO_TEST = [
    0.0,
    0.2,
    0.3,
]

# Se elimina batch size 8 y se prueban batch sizes más estables.
BATCH_SIZES_TO_TEST = [
    16,
    32,  # Si da error de memoria GPU, coméntalo.
]


# =========================================================
# CRITERIOS DE ESTABILIDAD DE VALIDATION LOSS
# =========================================================
# Se miran las últimas N epochs para detectar si la val_loss sube u oscila al final.
STABILITY_LAST_N_EPOCHS = 10

# Pendiente máxima permitida en la val_loss de las últimas epochs.
# Si es mayor que esto, se considera que la val_loss está subiendo al final.
STABILITY_SLOPE_MAX = 0.003

# Oscilación máxima permitida en las últimas epochs.
STABILITY_STD_MAX = 0.04

# Separación máxima permitida entre la última val_loss y la mínima val_loss.
# Se usa una condición absoluta y relativa para no penalizar diferencias muy pequeñas.
STABILITY_GAP_ABS_MAX = 0.08
STABILITY_GAP_RATIO_MAX = 0.20


# =========================================================
# CONFIGURACIÓN GPU OPCIONAL
# =========================================================
def configure_gpu_memory_growth():
    """
    Intenta evitar que TensorFlow reserve toda la memoria GPU al inicio.
    Si no se puede aplicar, el script continúa igualmente.
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            print(f"Memory growth activado para {len(gpus)} GPU(s).")
    except Exception as e:
        print(f"No se pudo activar memory growth: {e}")


# =========================================================
# FUNCIONES DE NOMBRES
# =========================================================
def float_to_name(x):
    """
    Convierte floats a texto seguro para nombres de carpetas.
    Ejemplo:
        1e-06 -> 1em06
        0.3   -> 0p3
    """
    if x == 0:
        return "0"

    if abs(x) < 1e-3:
        s = f"{x:.0e}"
    else:
        s = str(x)

    s = s.replace(".", "p")
    s = s.replace("-", "m")
    s = s.replace("+", "p")
    return s


def make_experiment_id(stage, learning_rate, dropout, batch_size, epochs):
    lr_name = float_to_name(learning_rate)
    dropout_name = float_to_name(dropout)

    return (
        f"{stage}"
        f"_lr{lr_name}"
        f"_dropout{dropout_name}"
        f"_bs{batch_size}"
        f"_ep{epochs}"
    )


# =========================================================
# MODELO
# =========================================================
def build_model_tuned(learning_rate, dropout):
    """
    ResNet50 ImageNet + GlobalAveragePooling2D + Dropout opcional + Dense sigmoid.
    """
    resnet_base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(base.IMAGE_SIZE, base.IMAGE_SIZE, 3),
    )

    # Fine tuning completo de ResNet50, igual que en tu script original.
    resnet_base.trainable = True

    inputs = layers.Input(shape=(base.IMAGE_SIZE, base.IMAGE_SIZE, 3))
    x = resnet_base(inputs)
    x = layers.GlobalAveragePooling2D()(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


# =========================================================
# ESTABILIDAD DE VALIDATION LOSS
# =========================================================
def compute_val_loss_stability(history_df):
    """
    Calcula métricas simples para decidir si la val_loss es estable.

    Un modelo se marca como inestable si:
        - la val_loss sube al final,
        - la val_loss oscila demasiado al final,
        - la última val_loss está claramente peor que la mínima val_loss.
    """
    val_loss = history_df["val_loss"].to_numpy(dtype=float)

    if len(val_loss) == 0:
        return {
            "mean_val_loss_last10": np.nan,
            "std_val_loss_last10": np.nan,
            "slope_val_loss_last10": np.nan,
            "last_val_loss_minus_min_val_loss": np.nan,
            "last_val_loss_minus_min_val_loss_ratio": np.nan,
            "val_loss_is_rising": True,
            "val_loss_is_oscillating": True,
            "val_loss_far_from_min": True,
            "is_stable_val_loss": False,
            "val_loss_stability_score": np.inf,
        }

    last_n = min(STABILITY_LAST_N_EPOCHS, len(val_loss))
    last_vals = val_loss[-last_n:]

    mean_last = float(np.mean(last_vals))
    std_last = float(np.std(last_vals))

    if last_n >= 2:
        x = np.arange(last_n, dtype=float)
        slope_last = float(np.polyfit(x, last_vals, 1)[0])
    else:
        slope_last = 0.0

    min_val_loss = float(np.min(val_loss))
    last_val_loss = float(val_loss[-1])
    gap_abs = float(last_val_loss - min_val_loss)

    if min_val_loss > 0:
        gap_ratio = float(gap_abs / min_val_loss)
    else:
        gap_ratio = np.nan

    val_loss_is_rising = slope_last > STABILITY_SLOPE_MAX
    val_loss_is_oscillating = std_last > STABILITY_STD_MAX
    val_loss_far_from_min = (
        gap_abs > STABILITY_GAP_ABS_MAX
        and gap_ratio > STABILITY_GAP_RATIO_MAX
    )

    is_stable = not (
        val_loss_is_rising
        or val_loss_is_oscillating
        or val_loss_far_from_min
    )

    # Score auxiliar: menor = más estable.
    slope_penalty = max(0.0, slope_last - STABILITY_SLOPE_MAX) / STABILITY_SLOPE_MAX
    std_penalty = max(0.0, std_last - STABILITY_STD_MAX) / STABILITY_STD_MAX

    if np.isfinite(gap_ratio):
        gap_penalty = max(0.0, gap_ratio - STABILITY_GAP_RATIO_MAX) / STABILITY_GAP_RATIO_MAX
    else:
        gap_penalty = 0.0

    stability_score = float(slope_penalty + std_penalty + gap_penalty)

    return {
        "mean_val_loss_last10": mean_last,
        "std_val_loss_last10": std_last,
        "slope_val_loss_last10": slope_last,
        "last_val_loss_minus_min_val_loss": gap_abs,
        "last_val_loss_minus_min_val_loss_ratio": gap_ratio,
        "val_loss_is_rising": bool(val_loss_is_rising),
        "val_loss_is_oscillating": bool(val_loss_is_oscillating),
        "val_loss_far_from_min": bool(val_loss_far_from_min),
        "is_stable_val_loss": bool(is_stable),
        "val_loss_stability_score": stability_score,
    }


# =========================================================
# PLOTS SIN TÍTULOS
# =========================================================
def plot_learning_curves_no_titles(history, experiment_id, results_dir):
    history_dict = history.history
    epochs_range = range(1, len(history_dict["loss"]) + 1)

    # Accuracy + loss juntas, sin títulos.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    if "accuracy" in history_dict and "val_accuracy" in history_dict:
        axes[0].plot(epochs_range, history_dict["accuracy"], label="train_accuracy")
        axes[0].plot(epochs_range, history_dict["val_accuracy"], label="val_accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_xlim(left=0)
        axes[0].set_ylim(bottom=0, top=1)
        axes[0].legend()

    axes[1].plot(epochs_range, history_dict["loss"], label="train_loss")
    axes[1].plot(epochs_range, history_dict["val_loss"], label="val_loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(results_dir, f"learning_curves_{experiment_id}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    # Accuracy por separado, sin título.
    if "accuracy" in history_dict and "val_accuracy" in history_dict:
        plt.figure(figsize=(4, 4))
        plt.plot(epochs_range, history_dict["accuracy"], label="train_accuracy")
        plt.plot(epochs_range, history_dict["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xlim(left=0)
        plt.ylim(bottom=0, top=1)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(results_dir, f"accuracy_curve_{experiment_id}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

    # Loss por separado, sin título.
    plt.figure(figsize=(4, 4))
    plt.plot(epochs_range, history_dict["loss"], label="train_loss")
    plt.plot(epochs_range, history_dict["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"loss_curve_{experiment_id}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_auc_history_no_title(history, experiment_id, results_dir):
    if "auc" not in history.history or "val_auc" not in history.history:
        return

    epochs_range = range(1, len(history.history["auc"]) + 1)

    plt.figure(figsize=(4, 4))
    plt.plot(epochs_range, history.history["auc"], label="train_auc")
    plt.plot(epochs_range, history.history["val_auc"], label="val_auc")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=1)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(results_dir, f"auc_curve_{experiment_id}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_single(metrics, split_name, experiment_id, results_dir):
    """
    Guarda una confusion matrix por separado, tamaño 4x4.
    Las labels del eje y se rotan 90 grados.
    """
    if metrics is None:
        return

    fig, ax = plt.subplots(figsize=(4, 4))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["cm"],
        display_labels=CLASS_NAMES,
    )
    disp.plot(ax=ax, cmap="Purples", colorbar=False, values_format="d")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("")

    ax.set_xticklabels(CLASS_NAMES, rotation=0)
    ax.set_yticklabels(CLASS_NAMES, rotation=90, va="center")
    ax.tick_params(axis="y", pad=8)

    for text in ax.texts:
        text.set_fontsize(14)

    plt.tight_layout()

    out_path = os.path.join(
        results_dir,
        f"confusion_matrix_{split_name.lower()}_{experiment_id}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc_from_metrics_no_title(metrics, split_name, experiment_id, results_dir):
    if metrics is None:
        return

    y_true = metrics["y_true"]
    y_prob = metrics["y_prob"]

    if len(np.unique(y_true)) < 2:
        print(f"No se puede dibujar ROC para {split_name}: solo hay una clase.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = os.path.join(
        results_dir,
        f"roc_curve_{split_name.lower()}_{experiment_id}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================================
# GUARDADO DE MÉTRICAS Y PREDICCIONES
# =========================================================
def flatten_metrics(metrics, prefix):
    if metrics is None:
        return {
            f"{prefix}_accuracy": np.nan,
            f"{prefix}_balanced_accuracy": np.nan,
            f"{prefix}_precision_macro": np.nan,
            f"{prefix}_recall_macro": np.nan,
            f"{prefix}_f1_macro": np.nan,
            f"{prefix}_auc": np.nan,
        }

    return {
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_balanced_accuracy": metrics["balanced_accuracy"],
        f"{prefix}_precision_macro": metrics["precision_macro"],
        f"{prefix}_recall_macro": metrics["recall_macro"],
        f"{prefix}_f1_macro": metrics["f1_macro"],
        f"{prefix}_auc": metrics["auc"],
    }


def save_classification_report(metrics, split_name, experiment_id, results_dir):
    if metrics is None:
        return

    report_df = pd.DataFrame(metrics["report"]).transpose()
    out_path = os.path.join(
        results_dir,
        f"classification_report_{split_name.lower()}_{experiment_id}.csv",
    )
    report_df.to_csv(out_path)


def save_predictions(pred_df, split_name, level_name, experiment_id, results_dir):
    if pred_df is None:
        return

    out_path = os.path.join(
        results_dir,
        f"{split_name.lower()}_predictions_{level_name}_{experiment_id}.csv",
    )
    pred_df.to_csv(out_path, index=False)


def add_empty_history_and_stability_columns(row):
    row["last_train_loss"] = np.nan
    row["last_val_loss"] = np.nan
    row["min_val_loss"] = np.nan
    row["epoch_min_val_loss"] = np.nan
    row["max_val_auc_history"] = np.nan
    row["epoch_max_val_auc_history"] = np.nan
    row["mean_val_loss_last10"] = np.nan
    row["std_val_loss_last10"] = np.nan
    row["slope_val_loss_last10"] = np.nan
    row["last_val_loss_minus_min_val_loss"] = np.nan
    row["last_val_loss_minus_min_val_loss_ratio"] = np.nan
    row["val_loss_is_rising"] = True
    row["val_loss_is_oscillating"] = True
    row["val_loss_far_from_min"] = True
    row["is_stable_val_loss"] = False
    row["val_loss_stability_score"] = np.inf
    return row


def write_summary(all_rows):
    summary_df = pd.DataFrame(all_rows)

    summary_path = os.path.join(
        TUNING_RESULTS_ROOT,
        f"grid_tuning_summary_{ACTIVE_INPUT_MODE}.csv",
    )
    summary_df.to_csv(summary_path, index=False)

    ok_df = summary_df[summary_df["status"] == "ok"].copy()

    if len(ok_df) > 0:
        ok_df["_stable_sort"] = ok_df["is_stable_val_loss"].fillna(False).astype(int)
        ok_df["_val_auc_sort"] = ok_df["val_auc"].fillna(-1)
        ok_df["_mean_loss_sort"] = ok_df["mean_val_loss_last10"].fillna(np.inf)
        ok_df["_std_loss_sort"] = ok_df["std_val_loss_last10"].fillna(np.inf)
        ok_df["_val_accuracy_sort"] = ok_df["val_accuracy"].fillna(-1)

        sorted_df = ok_df.sort_values(
            by=[
                "_stable_sort",
                "_val_auc_sort",
                "_mean_loss_sort",
                "_std_loss_sort",
                "_val_accuracy_sort",
            ],
            ascending=[False, False, True, True, False],
        ).drop(
            columns=[
                "_stable_sort",
                "_val_auc_sort",
                "_mean_loss_sort",
                "_std_loss_sort",
                "_val_accuracy_sort",
            ]
        )

        sorted_path = os.path.join(
            TUNING_RESULTS_ROOT,
            f"grid_tuning_summary_sorted_{ACTIVE_INPUT_MODE}.csv",
        )
        sorted_df.to_csv(sorted_path, index=False)


# =========================================================
# SELECCIÓN DEL MEJOR EXPERIMENTO
# =========================================================
def choose_best_grid(rows):
    """
    Elige el mejor experimento usando SOLO validation.

    Criterio:
        1. Se priorizan modelos con val_loss estable.
        2. Entre los modelos estables:
            - mayor val_auc,
            - menor mean_val_loss_last10,
            - menor std_val_loss_last10,
            - mayor val_accuracy.
        3. Si ningún modelo es estable, se elige el menos inestable y se deja indicado.
    """
    df = pd.DataFrame(rows)
    df = df[df["status"] == "ok"].copy()

    if df.empty:
        raise RuntimeError("No hay experimentos válidos en el grid tuning.")

    stable_df = df[df["is_stable_val_loss"] == True].copy()

    if len(stable_df) > 0:
        selection_pool = stable_df
        selected_from = "stable_models"
    else:
        selection_pool = df.copy()
        selected_from = "all_models_no_stable_model_found"

    selection_pool["_val_auc_sort"] = selection_pool["val_auc"].fillna(-1)
    selection_pool["_mean_loss_sort"] = selection_pool["mean_val_loss_last10"].fillna(np.inf)
    selection_pool["_std_loss_sort"] = selection_pool["std_val_loss_last10"].fillna(np.inf)
    selection_pool["_val_accuracy_sort"] = selection_pool["val_accuracy"].fillna(-1)
    selection_pool["_stability_score_sort"] = selection_pool["val_loss_stability_score"].fillna(np.inf)

    if selected_from == "stable_models":
        selection_pool = selection_pool.sort_values(
            by=[
                "_val_auc_sort",
                "_mean_loss_sort",
                "_std_loss_sort",
                "_val_accuracy_sort",
            ],
            ascending=[False, True, True, False],
        )
    else:
        selection_pool = selection_pool.sort_values(
            by=[
                "_stability_score_sort",
                "_val_auc_sort",
                "_mean_loss_sort",
                "_std_loss_sort",
                "_val_accuracy_sort",
            ],
            ascending=[True, False, True, True, False],
        )

    best = selection_pool.iloc[0].to_dict()
    best["selected_from"] = selected_from

    print("\n" + "-" * 100)
    print("MEJOR CONFIGURACIÓN DEL GRID")
    print("-" * 100)
    print(f"Selection pool:  {selected_from}")
    print(f"Experiment ID:   {best['experiment_id']}")
    print(f"Learning rate:   {best['learning_rate']}")
    print(f"Dropout:         {best['dropout']}")
    print(f"Batch size:      {best['batch_size']}")
    print(f"Epochs:          {best['epochs']}")
    print(f"Stable val_loss: {best['is_stable_val_loss']}")
    print(f"Val AUC:         {best['val_auc']:.4f}")
    print(f"Val accuracy:    {best['val_accuracy']:.4f}")
    print(f"Min val loss:    {best['min_val_loss']:.4f}")
    print(f"Mean loss last10:{best['mean_val_loss_last10']:.4f}")
    print(f"Std loss last10: {best['std_val_loss_last10']:.4f}")
    print(f"Slope last10:    {best['slope_val_loss_last10']:.6f}")
    print("-" * 100)

    return best


# =========================================================
# PREPARAR DATOS UNA SOLA VEZ
# =========================================================
def prepare_data_once():
    """
    Prepara datos una sola vez.

    Así todos los experimentos usan:
        - mismos sujetos train,
        - mismos sujetos validation,
        - mismos sujetos test,
        - mismas imágenes.
    """
    base.ACTIVE_INPUT_MODE = ACTIVE_INPUT_MODE

    if ACTIVE_INPUT_MODE not in base.INPUT_CONFIGS:
        raise ValueError(f"ACTIVE_INPUT_MODE no reconocido: {ACTIVE_INPUT_MODE}")

    config = base.INPUT_CONFIGS[ACTIVE_INPUT_MODE]
    title = config["title"]

    print("=" * 100)
    print(f"GRID TUNING PARA: {ACTIVE_INPUT_MODE} | {title}")
    print("=" * 100)

    df_metadata = base.read_subject_metadata()
    age_threshold = float(df_metadata["age"].mean())

    df_case = base.prepare_young_cu_vs_old_ad_from_csv(
        df_metadata,
        age_threshold,
    )

    split_df = base.make_fixed_subject_splits(df_case)

    samples_df = base.build_image_samples(config)

    samples_df = base.attach_labels_and_split(
        samples_df,
        df_case,
        split_df,
    )

    base.check_no_subject_leakage(samples_df)

    if samples_df.empty:
        raise ValueError("No hay muestras después de cruzar imágenes, CSV y split.")

    df_train = samples_df[samples_df["split"] == "train"].reset_index(drop=True)
    df_val = samples_df[samples_df["split"] == "val"].reset_index(drop=True)
    df_test = samples_df[samples_df["split"] == "test"].reset_index(drop=True)

    print(f"\nThreshold mean(AGE): {age_threshold:.4f}")
    print("Label 0 = young_CU | Label 1 = old_AD")

    print("\nMuestras:")
    print(f"Train: {len(df_train)} muestras | {df_train['subject_id'].nunique()} sujetos")
    print(f"Val:   {len(df_val)} muestras | {df_val['subject_id'].nunique()} sujetos")
    print(f"Test:  {len(df_test)} muestras | {df_test['subject_id'].nunique()} sujetos")

    # Guardar tablas generales.
    samples_df.to_csv(
        os.path.join(TUNING_RESULTS_ROOT, f"image_samples_{ACTIVE_INPUT_MODE}.csv"),
        index=False,
    )

    split_df.to_csv(
        os.path.join(TUNING_RESULTS_ROOT, f"subject_splits_{ACTIVE_INPUT_MODE}.csv"),
        index=False,
    )

    df_case.to_csv(
        os.path.join(TUNING_RESULTS_ROOT, f"subjects_case_{ACTIVE_INPUT_MODE}.csv"),
        index=False,
    )

    threshold_df = pd.DataFrame(
        [
            {
                "threshold_method": "mean",
                "age_threshold": age_threshold,
                "computed_from": "all_subjects_in_csv_with_valid_subject_age_dx",
                "n_subjects_for_threshold": len(df_metadata),
            }
        ]
    )

    threshold_df.to_csv(
        os.path.join(TUNING_RESULTS_ROOT, f"age_threshold_{ACTIVE_INPUT_MODE}.csv"),
        index=False,
    )

    return {
        "config": config,
        "title": title,
        "df_metadata": df_metadata,
        "df_case": df_case,
        "split_df": split_df,
        "samples_df": samples_df,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "age_threshold": age_threshold,
    }


# =========================================================
# EJECUTAR UN EXPERIMENTO
# =========================================================
def run_one_experiment(
    stage,
    learning_rate,
    dropout,
    batch_size,
    epochs,
    prepared_data,
    evaluate_test=False,
):
    experiment_id = make_experiment_id(
        stage=stage,
        learning_rate=learning_rate,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
    )

    results_dir = os.path.join(TUNING_RESULTS_ROOT, experiment_id)
    os.makedirs(results_dir, exist_ok=True)

    df_train = prepared_data["df_train"]
    df_val = prepared_data["df_val"]
    df_test = prepared_data["df_test"]
    age_threshold = prepared_data["age_threshold"]

    print("\n" + "=" * 100)
    print(f"EXPERIMENTO: {experiment_id}")
    print("=" * 100)
    print(f"Stage:         {stage}")
    print(f"Learning rate: {learning_rate}")
    print(f"Dropout:       {dropout}")
    print(f"Batch size:    {batch_size}")
    print(f"Epochs:        {epochs}")
    print(f"Evaluate test: {evaluate_test}")
    print(f"Resultados en: {results_dir}")

    row = {
        "status": "started",
        "experiment_id": experiment_id,
        "stage": stage,
        "input_mode": ACTIVE_INPUT_MODE,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "batch_size": batch_size,
        "epochs": epochs,
        "age_threshold": age_threshold,
        "n_train_subjects": df_train["subject_id"].nunique(),
        "n_val_subjects": df_val["subject_id"].nunique(),
        "n_test_subjects": df_test["subject_id"].nunique(),
        "n_train_samples": len(df_train),
        "n_val_samples": len(df_val),
        "n_test_samples": len(df_test),
        "results_dir": results_dir,
        "error": "",
    }

    try:
        # Actualizar globals del script original.
        # make_dataset usa base.BATCH_SIZE internamente.
        base.BATCH_SIZE = batch_size
        base.LEARNING_RATE = learning_rate
        base.EPOCHS = epochs
        base.ACTIVE_INPUT_MODE = ACTIVE_INPUT_MODE

        # Reproducibilidad.
        base.set_seed(base.RANDOM_STATE)

        # Crear datasets.
        ds_train = base.make_dataset(df_train, shuffle=True)
        ds_val = base.make_dataset(df_val, shuffle=False)

        # Limpiar antes de crear modelo.
        tf.keras.backend.clear_session()
        gc.collect()

        model = build_model_tuned(
            learning_rate=learning_rate,
            dropout=dropout,
        )

        # Sin EarlyStopping y sin ModelCheckpoint: siempre entrena las 60 epochs completas.
        history = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=epochs,
            verbose=1,
        )

        # =================================================
        # HISTORY
        # =================================================
        history_df = pd.DataFrame(history.history)
        history_df["epoch"] = np.arange(1, len(history_df) + 1)

        history_path = os.path.join(
            results_dir,
            f"training_history_{experiment_id}.csv",
        )
        history_df.to_csv(history_path, index=False)

        row["history_path"] = history_path
        row["last_train_loss"] = float(history_df["loss"].iloc[-1])
        row["last_val_loss"] = float(history_df["val_loss"].iloc[-1])
        row["min_val_loss"] = float(history_df["val_loss"].min())
        row["epoch_min_val_loss"] = int(
            history_df.loc[history_df["val_loss"].idxmin(), "epoch"]
        )

        if "val_auc" in history_df.columns:
            row["max_val_auc_history"] = float(history_df["val_auc"].max())
            row["epoch_max_val_auc_history"] = int(
                history_df.loc[history_df["val_auc"].idxmax(), "epoch"]
            )
        else:
            row["max_val_auc_history"] = np.nan
            row["epoch_max_val_auc_history"] = np.nan

        row.update(compute_val_loss_stability(history_df))

        # =================================================
        # PREDICCIONES TRAIN / VAL
        # =================================================
        train_sample_pred_df = base.predict_sample_level(model, df_train)
        val_sample_pred_df = base.predict_sample_level(model, df_val)

        train_subject_pred_df = base.aggregate_to_subject_level(train_sample_pred_df)
        val_subject_pred_df = base.aggregate_to_subject_level(val_sample_pred_df)

        train_metrics = base.evaluate_predictions(train_subject_pred_df, CLASS_NAMES)
        val_metrics = base.evaluate_predictions(val_subject_pred_df, CLASS_NAMES)

        # =================================================
        # TEST SOLO EN EL MODELO FINAL
        # =================================================
        if evaluate_test:
            test_sample_pred_df = base.predict_sample_level(model, df_test)
            test_subject_pred_df = base.aggregate_to_subject_level(test_sample_pred_df)
            test_metrics = base.evaluate_predictions(test_subject_pred_df, CLASS_NAMES)
        else:
            test_sample_pred_df = None
            test_subject_pred_df = None
            test_metrics = None

        # =================================================
        # GUARDAR REPORTS
        # =================================================
        save_classification_report(train_metrics, "train", experiment_id, results_dir)
        save_classification_report(val_metrics, "val", experiment_id, results_dir)
        save_classification_report(test_metrics, "test", experiment_id, results_dir)

        # =================================================
        # GUARDAR PREDICCIONES
        # =================================================
        save_predictions(train_sample_pred_df, "train", "sample_level", experiment_id, results_dir)
        save_predictions(val_sample_pred_df, "val", "sample_level", experiment_id, results_dir)
        save_predictions(test_sample_pred_df, "test", "sample_level", experiment_id, results_dir)

        save_predictions(train_subject_pred_df, "train", "subject_level", experiment_id, results_dir)
        save_predictions(val_subject_pred_df, "val", "subject_level", experiment_id, results_dir)
        save_predictions(test_subject_pred_df, "test", "subject_level", experiment_id, results_dir)

        # =================================================
        # PLOTS SIN TÍTULOS
        # =================================================
        plot_learning_curves_no_titles(history, experiment_id, results_dir)
        plot_auc_history_no_title(history, experiment_id, results_dir)

        plot_confusion_single(train_metrics, "Train", experiment_id, results_dir)
        plot_confusion_single(val_metrics, "Val", experiment_id, results_dir)

        plot_roc_from_metrics_no_title(val_metrics, "Val", experiment_id, results_dir)

        if evaluate_test and test_metrics is not None:
            plot_confusion_single(test_metrics, "Test", experiment_id, results_dir)
            plot_roc_from_metrics_no_title(test_metrics, "Test", experiment_id, results_dir)

        # =================================================
        # GUARDAR MODELO
        # =================================================
        model_path = os.path.join(results_dir, f"model_{experiment_id}.keras")
        model.save(model_path)
        row["model_path"] = model_path

        # =================================================
        # MÉTRICAS A SUMMARY
        # =================================================
        row.update(flatten_metrics(train_metrics, "train"))
        row.update(flatten_metrics(val_metrics, "val"))
        row.update(flatten_metrics(test_metrics, "test"))

        row["status"] = "ok"

        pd.DataFrame([row]).to_csv(
            os.path.join(results_dir, f"summary_results_{experiment_id}.csv"),
            index=False,
        )

        print("\nMÉTRICAS VALIDATION, subject-level")
        print(f"Val accuracy:         {val_metrics['accuracy']:.4f}")
        print(f"Val balanced acc:     {val_metrics['balanced_accuracy']:.4f}")
        print(f"Val precision macro:  {val_metrics['precision_macro']:.4f}")
        print(f"Val recall macro:     {val_metrics['recall_macro']:.4f}")
        print(f"Val F1 macro:         {val_metrics['f1_macro']:.4f}")
        print(f"Val AUC:              {val_metrics['auc']:.4f}")
        print(f"Min val loss:         {row['min_val_loss']:.4f} at epoch {row['epoch_min_val_loss']}")
        print(f"Mean val loss last10: {row['mean_val_loss_last10']:.4f}")
        print(f"Std val loss last10:  {row['std_val_loss_last10']:.4f}")
        print(f"Slope val loss last10:{row['slope_val_loss_last10']:.6f}")
        print(f"Stable val_loss:      {row['is_stable_val_loss']}")

        if evaluate_test and test_metrics is not None:
            print("\nMÉTRICAS TEST, subject-level")
            print(f"Test accuracy:        {test_metrics['accuracy']:.4f}")
            print(f"Test balanced acc:    {test_metrics['balanced_accuracy']:.4f}")
            print(f"Test precision macro: {test_metrics['precision_macro']:.4f}")
            print(f"Test recall macro:    {test_metrics['recall_macro']:.4f}")
            print(f"Test F1 macro:        {test_metrics['f1_macro']:.4f}")
            print(f"Test AUC:             {test_metrics['auc']:.4f}")

        # Limpieza.
        del model, ds_train, ds_val, history
        tf.keras.backend.clear_session()
        gc.collect()

        return row

    except Exception as e:
        print("\nERROR EN EXPERIMENTO:")
        print(experiment_id)
        print(str(e))

        error_text = traceback.format_exc()
        print(error_text)

        error_path = os.path.join(results_dir, f"error_{experiment_id}.txt")
        with open(error_path, "w") as f:
            f.write(error_text)

        row["status"] = "failed"
        row["error"] = str(e)
        row["error_path"] = error_path

        row.update(flatten_metrics(None, "train"))
        row.update(flatten_metrics(None, "val"))
        row.update(flatten_metrics(None, "test"))
        row = add_empty_history_and_stability_columns(row)
        row["model_path"] = ""

        pd.DataFrame([row]).to_csv(
            os.path.join(results_dir, f"summary_results_{experiment_id}.csv"),
            index=False,
        )

        tf.keras.backend.clear_session()
        gc.collect()

        return row


# =========================================================
# MAIN
# =========================================================
def main():
    configure_gpu_memory_growth()
    prepared_data = prepare_data_once()

    all_rows = []

    # =====================================================
    # GRID TUNING COMPLETO
    # =====================================================
    for learning_rate in LEARNING_RATES_TO_TEST:
        for dropout in DROPOUTS_TO_TEST:
            for batch_size in BATCH_SIZES_TO_TEST:
                row = run_one_experiment(
                    stage="grid",
                    learning_rate=learning_rate,
                    dropout=dropout,
                    batch_size=batch_size,
                    epochs=EPOCHS,
                    prepared_data=prepared_data,
                    evaluate_test=EVALUATE_TEST_FOR_TUNING,
                )

                all_rows.append(row)
                write_summary(all_rows)

    best_grid_row = choose_best_grid(all_rows)

    best_learning_rate = float(best_grid_row["learning_rate"])
    best_dropout = float(best_grid_row["dropout"])
    best_batch_size = int(best_grid_row["batch_size"])

    # =====================================================
    # MODELO FINAL
    # =====================================================
    # Se reentrena el modelo seleccionado con los mismos datos train/val y 60 epochs.
    # El test se evalúa solo aquí.
    final_row = run_one_experiment(
        stage="final_selected",
        learning_rate=best_learning_rate,
        dropout=best_dropout,
        batch_size=best_batch_size,
        epochs=EPOCHS,
        prepared_data=prepared_data,
        evaluate_test=True,
    )

    all_rows.append(final_row)
    write_summary(all_rows)

    # =====================================================
    # GUARDAR CONFIGURACIÓN FINAL
    # =====================================================
    final_config = pd.DataFrame(
        [
            {
                "best_learning_rate": best_learning_rate,
                "best_dropout": best_dropout,
                "best_batch_size": best_batch_size,
                "epochs_all_experiments": EPOCHS,
                "early_stopping_used": False,
                "model_checkpoint_used": False,
                "test_used_during_tuning": EVALUATE_TEST_FOR_TUNING,
                "selection_pool": best_grid_row.get("selected_from", "unknown"),
                "selection_criterion": (
                    "stable val_loss first; then highest val_auc; "
                    "then lowest mean_val_loss_last10; then lowest std_val_loss_last10; "
                    "then highest val_accuracy"
                ),
                "stability_last_n_epochs": STABILITY_LAST_N_EPOCHS,
                "stability_slope_max": STABILITY_SLOPE_MAX,
                "stability_std_max": STABILITY_STD_MAX,
                "stability_gap_abs_max": STABILITY_GAP_ABS_MAX,
                "stability_gap_ratio_max": STABILITY_GAP_RATIO_MAX,
                "reference_test_accuracy_previous_3_coronal_slices": REFERENCE_TEST_ACCURACY,
                "reference_test_auc_previous_3_coronal_slices": REFERENCE_TEST_AUC,
                "best_grid_experiment_id": best_grid_row["experiment_id"],
                "best_grid_results_dir": best_grid_row["results_dir"],
                "final_experiment_id": final_row["experiment_id"],
                "final_results_dir": final_row["results_dir"],
                "final_model_path": final_row["model_path"],
                "final_test_accuracy": final_row["test_accuracy"],
                "final_test_balanced_accuracy": final_row["test_balanced_accuracy"],
                "final_test_precision_macro": final_row["test_precision_macro"],
                "final_test_recall_macro": final_row["test_recall_macro"],
                "final_test_f1_macro": final_row["test_f1_macro"],
                "final_test_auc": final_row["test_auc"],
                "final_is_stable_val_loss": final_row["is_stable_val_loss"],
                "final_mean_val_loss_last10": final_row["mean_val_loss_last10"],
                "final_std_val_loss_last10": final_row["std_val_loss_last10"],
                "final_slope_val_loss_last10": final_row["slope_val_loss_last10"],
            }
        ]
    )

    final_config_path = os.path.join(
        TUNING_RESULTS_ROOT,
        f"best_grid_config_{ACTIVE_INPUT_MODE}.csv",
    )

    final_config.to_csv(final_config_path, index=False)

    print("\n" + "=" * 100)
    print("GRID TUNING TERMINADO")
    print("=" * 100)
    print(f"Mejor learning rate: {best_learning_rate}")
    print(f"Mejor dropout:       {best_dropout}")
    print(f"Mejor batch size:    {best_batch_size}")
    print(f"Epochs:              {EPOCHS}")
    print(f"EarlyStopping:       NO")
    print(f"Test durante tuning: {EVALUATE_TEST_FOR_TUNING}")
    print(f"\nConfiguración final guardada en:")
    print(final_config_path)
    print(f"\nResumen general guardado en:")
    print(os.path.join(TUNING_RESULTS_ROOT, f"grid_tuning_summary_{ACTIVE_INPUT_MODE}.csv"))
    print(f"\nResumen ordenado guardado en:")
    print(os.path.join(TUNING_RESULTS_ROOT, f"grid_tuning_summary_sorted_{ACTIVE_INPUT_MODE}.csv"))


if __name__ == "__main__":
    main()
