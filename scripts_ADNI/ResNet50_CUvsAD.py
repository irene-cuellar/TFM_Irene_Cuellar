#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet50 2.5D model for CU vs AD classification using ADNI coronal MRI slices.

This script trains a binary classification model using three coronal slices
(minus1, center and plus1) as the three input channels of a ResNet50 network.
The model is trained with a subject-level train, validation and test split.

The script saves the training history, train and test predictions, classification
metrics, confusion matrices, ROC-AUC curve and learning curves in the selected
results folder. Metric variability is estimated using bootstrap resampling on
the evaluated subjects.
"""


# =========================================================
# IMPORTS
# =========================================================
import os
import re
import gc
import random
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
# CONFIGURACIÓN
# =========================================================
CSV_PATH = "/pool/home/AD_Multimodal/ADNI/irene_adni/subjects_tfm_irene_filtered_all.csv"

SUBJECT_COL = "subject"
LABEL_COL = "DX_binary"

LABEL_MAP = {"CU": 0, "AD": 1}
CLASS_NAMES = ["CU", "AD"]
CLASS_LABELS = [0, 1]

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 1e-6
RANDOM_STATE = 42
VAL_SIZE_FROM_TRAIN = 0.20  # 80% train_full -> 64% train y 16% val del total
BOOTSTRAP_ITERATIONS = 1000

MODES = {
    "crop": {
        "center": "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_central_crop_final",
        "minus1": "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_minus1_crop_final",
        "plus1": "/pool/home/AD_Multimodal/ADNI/irene_adni/coronal/coronal_plus1_crop_final",
    },
}

PLOT_FIGURES = True
SAVE_PREDICTIONS_CSV = True
RESULTS_DIR = "/pool/home/AD_Multimodal/ADNI/irene_adni/results_def"

plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


# =========================================================
# FUNCIONES GENERALES
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def extraer_subject_id(nombre_archivo):
    match = re.search(r"(sub-[A-Za-z0-9]+)", nombre_archivo)
    return match.group(1) if match else None


def normalizar_subject_id(x):
    if pd.isna(x):
        return None
    return str(x).strip()


def normalizar_label(x):
    x = str(x).strip().upper()
    return x if x in LABEL_MAP else None


def save_figure(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {out_path}")


def save_dataframe(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"CSV guardado en: {out_path}")


def build_dataframe_triplets(mode_dirs):
    """Crea un dataframe con una fila por sujeto y sus 3 rutas de imagen."""
    center_dir = mode_dirs["center"]
    minus1_dir = mode_dirs["minus1"]
    plus1_dir = mode_dirs["plus1"]

    df_csv = pd.read_csv(CSV_PATH, low_memory=False)

    df_labels = df_csv[[SUBJECT_COL, LABEL_COL]].copy()
    df_labels["subject_id"] = df_labels[SUBJECT_COL].apply(normalizar_subject_id)
    df_labels["label_name"] = df_labels[LABEL_COL].apply(normalizar_label)
    df_labels = df_labels.dropna(subset=["subject_id", "label_name"])
    df_labels = df_labels.drop_duplicates(subset=["subject_id"])
    df_labels["label"] = df_labels["label_name"].map(LABEL_MAP)

    label_dict = df_labels.set_index("subject_id")[["label_name", "label"]].to_dict("index")

    center_files = {f for f in os.listdir(center_dir) if f.lower().endswith(".png")}
    minus1_files = {f for f in os.listdir(minus1_dir) if f.lower().endswith(".png")}
    plus1_files = {f for f in os.listdir(plus1_dir) if f.lower().endswith(".png")}

    common_files = sorted(center_files & minus1_files & plus1_files)
    if len(common_files) == 0:
        raise ValueError("No hay archivos comunes entre center/minus1/plus1")

    rows = []
    for filename in common_files:
        subject_id = extraer_subject_id(filename)
        if subject_id is None or subject_id not in label_dict:
            continue

        rows.append(
            {
                "filename": filename,
                "subject_id": subject_id,
                "path_minus1": os.path.join(minus1_dir, filename),
                "path_center": os.path.join(center_dir, filename),
                "path_plus1": os.path.join(plus1_dir, filename),
                "label_name": label_dict[subject_id]["label_name"],
                "label": label_dict[subject_id]["label"],
            }
        )

    df_final = pd.DataFrame(rows)

    if df_final.empty:
        raise ValueError("El dataframe final está vacío. Revisa nombres y etiquetas.")

    duplicados = df_final["subject_id"].duplicated().sum()
    if duplicados > 0:
        raise ValueError(f"Hay {duplicados} sujetos con más de una tripleta en este modo")

    return df_final.sort_values("subject_id").reset_index(drop=True)


@tf.autograph.experimental.do_not_convert  # para que no salgan los warnings
def load_triplet(path_minus1, path_center, path_plus1, label):
    img_m1 = tf.io.read_file(path_minus1)
    img_m1 = tf.image.decode_png(img_m1, channels=1)
    img_m1 = tf.image.resize(img_m1, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
    img_m1 = tf.cast(img_m1, tf.float32)

    img_c = tf.io.read_file(path_center)
    img_c = tf.image.decode_png(img_c, channels=1)
    img_c = tf.image.resize(img_c, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
    img_c = tf.cast(img_c, tf.float32)

    img_p1 = tf.io.read_file(path_plus1)
    img_p1 = tf.image.decode_png(img_p1, channels=1)
    img_p1 = tf.image.resize(img_p1, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
    img_p1 = tf.cast(img_p1, tf.float32)

    # Las 3 slices se apilan como si fueran los 3 canales de una imagen.
    img = tf.concat([img_m1, img_c, img_p1], axis=-1)
    img = preprocess_input(img)
    return img, label


def make_dataset(df, shuffle=False):
    """Construye el tf.data.Dataset a partir de las 3 imágenes por sujeto."""
    paths_minus1 = df["path_minus1"].values
    paths_center = df["path_center"].values
    paths_plus1 = df["path_plus1"].values
    labels = df["label"].astype("float32").values

    ds = tf.data.Dataset.from_tensor_slices((paths_minus1, paths_center, paths_plus1, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_STATE, reshuffle_each_iteration=True)

    ds = ds.map(load_triplet, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# =========================================================
# MODEL
# =========================================================
def build_model():
    """
    Baseline simple:
    - arquitectura ResNet50
    - sin congelar capas
    - solo se adapta la salida a clasificación binaria
    """
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
# EVALUACIÓN
# =========================================================
def predict_probabilities(model, df):
    ds = make_dataset(df, shuffle=False)
    y_true = df["label"].values.astype(int)
    y_prob = model.predict(ds, verbose=0).ravel()
    return y_true, y_prob


def bootstrap_metric_std(y_true, y_prob, threshold=0.5, n_bootstrap=1000, seed=42):
    """
    Calcula la desviación estándar de las métricas con bootstrap sobre sujetos.

    Nota: como este script usa un único split y no K-Fold, esta std no es entre folds.
    Es una estimación por remuestreo del conjunto evaluado.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    rng = np.random.default_rng(seed)
    n = len(y_true)

    values = {
        "accuracy": [],
        "precision_macro": [],
        "recall_macro": [],
        "f1_macro": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]

        values["accuracy"].append(accuracy_score(yt, yp))
        values["precision_macro"].append(
            precision_score(yt, yp, labels=CLASS_LABELS, average="macro", zero_division=0)
        )
        values["recall_macro"].append(
            recall_score(yt, yp, labels=CLASS_LABELS, average="macro", zero_division=0)
        )
        values["f1_macro"].append(
            f1_score(yt, yp, labels=CLASS_LABELS, average="macro", zero_division=0)
        )

    return {f"{metric}_std": float(np.std(metric_values, ddof=1)) for metric, metric_values in values.items()}


def evaluate_split(y_true, y_prob, threshold=0.5, bootstrap_seed=42):
    y_pred = (y_prob >= threshold).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),
        "auc": auc,
        "cm": cm,
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    metrics.update(
        bootstrap_metric_std(
            y_true,
            y_prob,
            threshold=threshold,
            n_bootstrap=BOOTSTRAP_ITERATIONS,
            seed=bootstrap_seed,
        )
    )

    return metrics


def metrics_to_row(mode_name, split_name, metrics):
    return {
        "mode": mode_name,
        "split": split_name,
        "n_subjects": len(metrics["y_true"]),
        "accuracy": metrics["accuracy"],
        "accuracy_std": metrics["accuracy_std"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision_macro": metrics["precision_macro"],
        "precision_macro_std": metrics["precision_macro_std"],
        "recall_macro": metrics["recall_macro"],
        "recall_macro_std": metrics["recall_macro_std"],
        "f1_macro": metrics["f1_macro"],
        "f1_macro_std": metrics["f1_macro_std"],
        "auc": metrics["auc"],
    }


def make_predictions_dataframe(df_split, metrics, mode_name, split_name):
    pred_df = df_split[["subject_id", "label_name", "label"]].copy()
    pred_df["y_prob"] = metrics["y_prob"]
    pred_df["y_pred"] = metrics["y_pred"]
    pred_df["pred_label_name"] = pred_df["y_pred"].map({0: "CU", 1: "AD"})
    pred_df["correct"] = (pred_df["label"] == pred_df["y_pred"]).astype(int)
    pred_df["mode"] = mode_name
    pred_df["split"] = split_name
    return pred_df


# =========================================================
# PLOTS
# =========================================================
def plot_learning_curves(history, mode_name, results_dir):
    epochs_range = range(1, len(history.history["loss"]) + 1)

    # Loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs_range, history.history["loss"], label="train_loss")
    ax.plot(epochs_range, history.history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, os.path.join(results_dir, f"learning_curve_loss_resnet50_25D_{mode_name}.png"))

    # Accuracy curve
    if "accuracy" in history.history and "val_accuracy" in history.history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs_range, history.history["accuracy"], label="train_accuracy")
        ax.plot(epochs_range, history.history["val_accuracy"], label="val_accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, os.path.join(results_dir, f"learning_curve_accuracy_resnet50_25D_{mode_name}.png"))


def plot_confusion_matrix_separate(y_true, y_pred, split_name, mode_name, results_dir):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Purples", colorbar=False, values_format="d")

    # Sin título
    ax.set_title("")

    # Labels del eje y rotadas 90 grados y centradas
    ax.set_yticklabels(CLASS_NAMES, rotation=90, va="center")
    ax.tick_params(axis="y", pad=8)

    for text in ax.texts:
        text.set_fontsize(14)

    fig.tight_layout()
    out_path = os.path.join(results_dir, f"confusion_matrix_{split_name}_resnet50_25D_{mode_name}.png")
    save_figure(fig, out_path)


def plot_roc_curve(y_true, y_prob, mode_name, results_dir):
    if len(np.unique(y_true)) < 2:
        print("No se puede dibujar la ROC curve porque solo hay una clase en test.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = os.path.join(results_dir, f"roc_auc_test_resnet50_25D_{mode_name}.png")
    save_figure(fig, out_path)


# =========================================================
# GUARDAR RESULTADOS
# =========================================================
def save_history(history, mode_name, results_dir):
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    save_dataframe(history_df, os.path.join(results_dir, f"history_resnet50_25D_{mode_name}.csv"))


def save_classification_report(report, mode_name, split_name, results_dir):
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "class_or_metric"})
    save_dataframe(
        report_df,
        os.path.join(results_dir, f"classification_report_{split_name}_resnet50_25D_{mode_name}.csv"),
    )


def save_confusion_matrix_csv(cm, mode_name, split_name, results_dir):
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.index.name = "true_label"
    out_path = os.path.join(results_dir, f"confusion_matrix_{split_name}_resnet50_25D_{mode_name}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm_df.to_csv(out_path)
    print(f"CSV guardado en: {out_path}")


def save_split_subjects(df_train, df_val, df_test, mode_name, results_dir):
    split_df = pd.concat(
        [
            df_train[["subject_id", "label_name", "label"]].assign(split="train"),
            df_val[["subject_id", "label_name", "label"]].assign(split="val"),
            df_test[["subject_id", "label_name", "label"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    split_df["mode"] = mode_name
    save_dataframe(split_df, os.path.join(results_dir, f"split_subjects_resnet50_25D_{mode_name}.csv"))


# =========================================================
# TRAIN / VAL / TEST SPLIT
# =========================================================
def run_single_split_for_mode(mode_name, df_mode):
    print("\n" + "=" * 90)
    print(f"MODE: {mode_name}")
    print("=" * 90)

    subjects_df = df_mode[["subject_id", "label", "label_name"]].drop_duplicates().reset_index(drop=True)

    print("Número total de sujetos:", len(subjects_df))
    print("Distribución total:")
    print(subjects_df["label_name"].value_counts())

    # Primer split: train_full / test
    train_full_subjects, test_subjects = train_test_split(
        subjects_df,
        test_size=0.20,
        stratify=subjects_df["label"],
        random_state=RANDOM_STATE,
    )

    # Segundo split: train / val dentro de train_full
    train_subjects, val_subjects = train_test_split(
        train_full_subjects,
        test_size=VAL_SIZE_FROM_TRAIN,
        stratify=train_full_subjects["label"],
        random_state=RANDOM_STATE,
    )

    train_ids = set(train_subjects["subject_id"])
    val_ids = set(val_subjects["subject_id"])
    test_ids = set(test_subjects["subject_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    df_train = df_mode[df_mode["subject_id"].isin(train_ids)].sort_values("subject_id").reset_index(drop=True)
    df_val = df_mode[df_mode["subject_id"].isin(val_ids)].sort_values("subject_id").reset_index(drop=True)
    df_test = df_mode[df_mode["subject_id"].isin(test_ids)].sort_values("subject_id").reset_index(drop=True)

    print("\nTrain:", len(df_train), "| Val:", len(df_val), "| Test:", len(df_test))
    print("Distribución train:")
    print(df_train["label_name"].value_counts())
    print("Distribución val:")
    print(df_val["label_name"].value_counts())
    print("Distribución test:")
    print(df_test["label_name"].value_counts())

    save_split_subjects(df_train, df_val, df_test, mode_name, RESULTS_DIR)

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

    save_history(history, mode_name, RESULTS_DIR)

    # Predicciones para train y test
    y_train_true, y_train_prob = predict_probabilities(model, df_train)
    y_test_true, y_test_prob = predict_probabilities(model, df_test)

    metrics_train = evaluate_split(
        y_train_true,
        y_train_prob,
        threshold=0.5,
        bootstrap_seed=RANDOM_STATE,
    )
    metrics_test = evaluate_split(
        y_test_true,
        y_test_prob,
        threshold=0.5,
        bootstrap_seed=RANDOM_STATE + 1,
    )

    print("\nMÉTRICAS TEST")
    print(f"Accuracy:        {metrics_test['accuracy']:.4f} ± {metrics_test['accuracy_std']:.4f}")
    print(f"Balanced acc:    {metrics_test['balanced_accuracy']:.4f}")
    print(f"Precision macro: {metrics_test['precision_macro']:.4f} ± {metrics_test['precision_macro_std']:.4f}")
    print(f"Recall macro:    {metrics_test['recall_macro']:.4f} ± {metrics_test['recall_macro_std']:.4f}")
    print(f"F1 macro:        {metrics_test['f1_macro']:.4f} ± {metrics_test['f1_macro_std']:.4f}")
    print(f"AUC:             {metrics_test['auc']:.4f}")

    print("\nClassification report test:")
    print(pd.DataFrame(metrics_test["report"]).transpose())

    print("\nConfusion matrix test:")
    print(metrics_test["cm"])

    if PLOT_FIGURES:
        plot_learning_curves(history, mode_name, RESULTS_DIR)
        plot_confusion_matrix_separate(
            metrics_train["y_true"],
            metrics_train["y_pred"],
            "train",
            mode_name,
            RESULTS_DIR,
        )
        plot_confusion_matrix_separate(
            metrics_test["y_true"],
            metrics_test["y_pred"],
            "test",
            mode_name,
            RESULTS_DIR,
        )
        plot_roc_curve(metrics_test["y_true"], metrics_test["y_prob"], mode_name, RESULTS_DIR)

    train_pred_df = make_predictions_dataframe(df_train, metrics_train, mode_name, "train")
    test_pred_df = make_predictions_dataframe(df_test, metrics_test, mode_name, "test")

    if SAVE_PREDICTIONS_CSV:
        save_dataframe(
            train_pred_df,
            os.path.join(RESULTS_DIR, f"train_predictions_resnet50_25D_{mode_name}.csv"),
        )
        save_dataframe(
            test_pred_df,
            os.path.join(RESULTS_DIR, f"test_predictions_resnet50_25D_{mode_name}.csv"),
        )

    # Guardar métricas, reports y matrices en CSV
    metrics_df = pd.DataFrame(
        [
            metrics_to_row(mode_name, "train", metrics_train),
            metrics_to_row(mode_name, "test", metrics_test),
        ]
    )
    save_dataframe(metrics_df, os.path.join(RESULTS_DIR, f"metrics_resnet50_25D_{mode_name}.csv"))

    save_classification_report(metrics_train["report"], mode_name, "train", RESULTS_DIR)
    save_classification_report(metrics_test["report"], mode_name, "test", RESULTS_DIR)

    save_confusion_matrix_csv(metrics_train["cm"], mode_name, "train", RESULTS_DIR)
    save_confusion_matrix_csv(metrics_test["cm"], mode_name, "test", RESULTS_DIR)

    del model, ds_train, ds_val, history
    tf.keras.backend.clear_session()
    gc.collect()

    return metrics_train, metrics_test, train_pred_df, test_pred_df


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(RANDOM_STATE)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    for mode_name, mode_dirs in MODES.items():
        print("\nLeyendo dataframe...")
        df_mode = build_dataframe_triplets(mode_dirs)

        metrics_train, metrics_test, train_pred_df, test_pred_df = run_single_split_for_mode(mode_name, df_mode)
        all_results[mode_name] = {
            "metrics_train": metrics_train,
            "metrics_test": metrics_test,
            "train_pred_df": train_pred_df,
            "test_pred_df": test_pred_df,
        }

    print("\n" + "=" * 90)
    print("RESUMEN FINAL")
    print("=" * 90)

    summary_rows = []
    for mode_name, result in all_results.items():
        metrics_test = result["metrics_test"]
        test_pred_df = result["test_pred_df"]

        print(f"\nModo:             {mode_name}")
        print(f"Sujetos test:     {len(test_pred_df)}")
        print(f"Accuracy test:    {metrics_test['accuracy']:.4f} ± {metrics_test['accuracy_std']:.4f}")
        print(f"Precision test:   {metrics_test['precision_macro']:.4f} ± {metrics_test['precision_macro_std']:.4f}")
        print(f"Recall test:      {metrics_test['recall_macro']:.4f} ± {metrics_test['recall_macro_std']:.4f}")
        print(f"F1 macro test:    {metrics_test['f1_macro']:.4f} ± {metrics_test['f1_macro_std']:.4f}")
        print(f"AUC test:         {metrics_test['auc']:.4f}")

        summary_rows.append(metrics_to_row(mode_name, "test", metrics_test))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(RESULTS_DIR, "summary_resnet50_25D_crop.csv")
    save_dataframe(summary_df, summary_csv)

    print("\nTodos los resultados se han guardado en:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
