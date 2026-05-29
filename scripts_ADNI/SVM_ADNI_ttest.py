#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM ADNI - resultados finales con 5-fold cross-validation

Este script ejecuta dos problemas de clasificación con SVM lineal:

1) CU vs AD
   - No requiere AGE.
   - Solo elimina sujetos con NaN en alguna variable volumétrica usada.

2) young_CU vs old_AD
   - young_CU = CU con AGE <= media de AGE de todos los sujetos
   - old_AD   = AD con AGE >= media de AGE de todos los sujetos
   - Sí requiere AGE para definir los grupos.
   - Después elimina sujetos con NaN en alguna variable volumétrica usada.

En ambos casos se usan las 6 variables originales, excluyendo ICV:
    Ventricles, Hippocampus, WholeBrain, Entorhinal, Fusiform, MidTemp

Se guarda todo en:
    /pool/home/AD_Multimodal/ADNI/irene_adni/results_SVM_def

Incluye:
    - 5-fold cross-validation estratificada.
    - Métricas por fold.
    - Mean y std de Accuracy, Precision, Recall, F1-score y ROC-AUC.
    - Matriz de confusión global con predicciones out-of-fold.
    - Curva ROC global con predicciones out-of-fold.
    - Histogramas de distribuciones en 3 columnas x 2 filas.
    - T-test independiente de cada feature entre los dos grupos de cada problema.
    - Corrección FDR Benjamini-Hochberg para los p-values de las 6 features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# =============================================================================
# 1. Rutas
# =============================================================================
BASE_DIR = Path("/pool/home/AD_Multimodal/ADNI/irene_adni")
CSV_PATH = BASE_DIR / "subjects_tfm_ADNI_all.csv"

RESULTS_DIR = BASE_DIR / "results_SVM_def_29may"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("\nCSV de entrada:")
print(CSV_PATH)

print("\nCarpeta principal de resultados:")
print(RESULTS_DIR)


# =============================================================================
# 2. Configuración
# =============================================================================
TARGET_COL = "DX_binary"
AGE_COL = "AGE"

FEATURES = [
    "Ventricles",
    "Hippocampus",
    "WholeBrain",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
]

N_SPLITS = 5
RANDOM_STATE = 42

# Configuración de histogramas
HIST_N_BINS = 20
HIST_YTICK_STEP = 20
HIST_Y_MARGIN_FACTOR = 1.08
HIST_MAX_X_TICKS = 5

def get_fixed_hist_ymax(problem_name):
    if problem_name == "CU vs AD":
        return 100
    elif problem_name == "young CU vs old AD":
        return 70
    else:
        return None

# Tamaños de letra
TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 15
CM_NUMBER_FONTSIZE = 14


# =============================================================================
# 3. Funciones auxiliares
# =============================================================================
def validate_columns(df, columns):
    """Comprueba que existan las columnas necesarias."""
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas necesarias en el CSV: {missing}")


def save_figure(fig, output_path):
    """Guarda una figura y la cierra."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figura guardada en: {output_path}")

    plt.close(fig)


def make_task_suffix(problem_name):
    """
    Convierte el nombre de la tarea en un sufijo seguro para archivos.

    Ejemplos:
        "CU vs AD" -> "cu_vs_ad"
        "young CU vs old AD" -> "young_cu_vs_old_ad"
    """
    suffix = str(problem_name).strip().lower()

    replacements = {
        " ": "_",
        "-": "_",
        "/": "_",
        "\\": "_",
        ":": "",
        "(": "",
        ")": "",
        ",": "",
        ".": "",
    }

    for old, new in replacements.items():
        suffix = suffix.replace(old, new)

    while "__" in suffix:
        suffix = suffix.replace("__", "_")

    return suffix.strip("_")


def get_metadata_columns(df):
    """Selecciona columnas informativas si existen."""
    preferred_cols = [
        "subject",
        "session",
        "registration_final",
        "EXAMDATE",
        "DX_bl",
        "DX",
        TARGET_COL,
        AGE_COL,
        "PTGENDER",
        "MMSE",
        "source_dataset",
        "group_age_dx",
        "age_threshold_value",
    ]

    return [col for col in preferred_cols if col in df.columns]


def make_model():
    """Crea el pipeline completo para evitar leakage dentro de cada fold."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", random_state=RANDOM_STATE)),
        ]
    )


def make_metrics_row(
    y_true,
    y_pred,
    scores,
    problem_name,
    fold,
    split_name,
):
    """
    Métricas principales.

    La clase positiva es siempre 1:
        - AD en CU vs AD
        - old_AD en young_CU vs old_AD
    """
    return {
        "problem": problem_name,
        "fold": fold,
        "split": split_name,

        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_true, scores),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "precision_weighted": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "n_total": len(y_true),
        "n_class_0": int((np.asarray(y_true) == 0).sum()),
        "n_class_1": int((np.asarray(y_true) == 1).sum()),
    }


def summarize_cv_metrics(metrics_by_fold):
    """
    Calcula mean y std de las métricas de validación de los 5 folds.
    """
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]

    rows = []

    for problem_name, df_problem in metrics_by_fold.groupby("problem"):
        df_valid = df_problem[df_problem["split"] == "validation"].copy()

        row = {
            "problem": problem_name,
            "n_folds": df_valid["fold"].nunique(),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = df_valid[col].mean()
            row[f"{col}_std"] = df_valid[col].std(ddof=1)

        rows.append(row)

    return pd.DataFrame(rows)


def make_classification_report_df(
    y_true,
    y_pred,
    labels_numeric,
    labels_display,
    problem_name,
):
    """
    Devuelve el classification report out-of-fold en formato tabla.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_numeric,
        target_names=labels_display,
        zero_division=0,
        output_dict=True,
    )

    rows = []

    for label, values in report.items():

        if isinstance(values, dict):
            rows.append(
                {
                    "problem": problem_name,
                    "label": label,
                    "precision": values.get("precision", np.nan),
                    "recall": values.get("recall", np.nan),
                    "f1_score": values.get("f1-score", np.nan),
                    "support": values.get("support", np.nan),
                }
            )

        else:
            rows.append(
                {
                    "problem": problem_name,
                    "label": label,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1_score": values,
                    "support": len(y_true),
                }
            )

    return pd.DataFrame(rows)


def plot_confusion_matrix_cv(
    cm_cv,
    labels_display,
    problem_name,
    output_dir,
):
    """
    Guarda la matriz de confusión usando predicciones out-of-fold.
    """
    task_suffix = make_task_suffix(problem_name)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_cv,
        display_labels=labels_display,
    )

    disp.plot(
        ax=ax,
        cmap="Purples",
        colorbar=False,
    )

    ax.set_xlabel("Predicted label", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("True label", fontsize=AXIS_LABEL_FONTSIZE)

    ax.set_xticklabels(
        labels_display,
        fontsize=TICK_LABEL_FONTSIZE,
    )

    ax.set_yticklabels(
        labels_display,
        rotation=90,
        va="center",
        ha="center",
        fontsize=TICK_LABEL_FONTSIZE,
    )

    ax.tick_params(axis="y", pad=8)

    for text in disp.text_.ravel():
        text.set_fontsize(CM_NUMBER_FONTSIZE)

    plt.tight_layout()

    save_figure(
        fig,
        output_dir / f"confusion_matrix_5fold_cv_{task_suffix}.png",
    )


def get_common_hist_ymax(axes):
    """
    Calcula automáticamente un ymax común para todos los histogramas.

    Tiene en cuenta:
        - la altura de las barras
        - la altura de las curvas KDE

    Así evita que se corte alguna distribución.
    """
    max_seen = 0.0

    for ax in axes:

        for patch in ax.patches:
            max_seen = max(max_seen, float(patch.get_height()))

        for line in ax.lines:
            y_data = np.asarray(line.get_ydata(), dtype=float)

            if y_data.size > 0:
                max_seen = max(max_seen, float(np.nanmax(y_data)))

    if max_seen <= 0 or np.isnan(max_seen):
        return HIST_YTICK_STEP

    ymax_with_margin = max_seen * HIST_Y_MARGIN_FACTOR

    ymax = int(
        np.ceil(ymax_with_margin / HIST_YTICK_STEP) * HIST_YTICK_STEP
    )

    return max(ymax, HIST_YTICK_STEP)


def plot_histograms_all_variables(
    X,
    y,
    labels_display,
    problem_name,
    output_dir,
):
    """
    Histogramas superpuestos de todas las variables en volumen original.

    Cambios aplicados:
        - 3 columnas y 2 filas.
        - Todas las variables tienen el mismo ymin = 0.
        - Todas las variables tienen el mismo ymax calculado automáticamente.
        - El ymax se calcula teniendo en cuenta barras y curvas KDE.
        - En el eje X hay máximo 5 ticks.
        - En el eje Y solo aparecen ticks y números en las gráficas de la izquierda.
        - La label del eje Y solo aparece en las gráficas de la izquierda.
        - La label del eje X solo aparece en las gráficas inferiores.
        - Letras y números más grandes.
    """
    task_suffix = make_task_suffix(problem_name)

    df_plot = X.copy()
    df_plot["group"] = y.values

    n_vars = len(X.columns)

    ncols = 3
    nrows = 2

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, 8.5),
    )

    axes = np.array(axes).reshape(-1)

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_0 = default_colors[0]
    color_1 = default_colors[1]

    legend_handles = [
        Patch(
            facecolor=color_0,
            edgecolor="none",
            label=labels_display[0],
        ),
        Patch(
            facecolor=color_1,
            edgecolor="none",
            label=labels_display[1],
        ),
    ]

    # -------------------------------------------------------------------------
    # Primero se dibujan todas las gráficas.
    # Después se calcula el ymax común automáticamente.
    # -------------------------------------------------------------------------
    for i, var in enumerate(X.columns):

        ax = axes[i]

        vals_0 = (
            df_plot
            .loc[df_plot["group"] == 0, var]
            .dropna()
            .astype(float)
        )

        vals_1 = (
            df_plot
            .loc[df_plot["group"] == 1, var]
            .dropna()
            .astype(float)
        )

        combined_vals = pd.concat(
            [vals_0, vals_1],
            axis=0,
        )

        ax.set_title(var, fontsize=TITLE_FONTSIZE)

        if combined_vals.empty:
            ax.set_xlim(0, 1)
            continue

        min_value = combined_vals.min()
        max_value = combined_vals.max()

        if min_value == max_value:
            x_min = min_value - 1
            x_max = max_value + 1
            bins = HIST_N_BINS
            bin_width = 1

        else:
            x_margin = (max_value - min_value) * 0.05

            x_min = min_value - x_margin
            x_max = max_value + x_margin

            bins = np.linspace(
                x_min,
                x_max,
                HIST_N_BINS + 1,
            )

            bin_width = bins[1] - bins[0]

        ax.hist(
            vals_0,
            bins=bins,
            alpha=0.25,
            color=color_0,
            label=labels_display[0],
        )

        ax.hist(
            vals_1,
            bins=bins,
            alpha=0.25,
            color=color_1,
            label=labels_display[1],
        )

        for vals, color in [
            (vals_0, color_0),
            (vals_1, color_1),
        ]:

            vals_array = vals.to_numpy(dtype=float)

            if len(vals_array) > 1 and np.std(vals_array) > 0:

                x_grid = np.linspace(
                    vals_array.min(),
                    vals_array.max(),
                    300,
                )

                kde = gaussian_kde(vals_array)

                y_grid = kde(x_grid) * len(vals_array) * bin_width

                ax.plot(
                    x_grid,
                    y_grid,
                    linewidth=2.5,
                    color=color,
                    alpha=1.0,
                )

        ax.set_xlim(x_min, x_max)

        # Máximo 5 ticks en el eje X.
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=HIST_MAX_X_TICKS)
        )

    # Eliminar ejes sobrantes si existen.
    for j in range(n_vars, len(axes)):
        fig.delaxes(axes[j])

    # Calcular ymax común automático después de dibujar barras y curvas.
    valid_axes = axes[:n_vars]
    fixed_ymax = get_fixed_hist_ymax(problem_name)
    
    if fixed_ymax is not None:
        hist_ymax = fixed_ymax
    else:
        hist_ymax = get_common_hist_ymax(valid_axes)

    # -------------------------------------------------------------------------
    # Aplicar formato final común.
    # -------------------------------------------------------------------------
    for i, ax in enumerate(valid_axes):

        is_left_column = (i % ncols == 0)
        is_bottom_row = (i >= ncols)

        # Mismo ymin/ymax para todos.
        ax.set_ylim(0, hist_ymax)
        ax.set_yticks(
            np.arange(
                0,
                hist_ymax + 1,
                HIST_YTICK_STEP,
            )
        )

        # Label del eje Y solo en las gráficas de la izquierda.
        if is_left_column:
            ax.set_ylabel("Count", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_ylabel("")

        # Ticks y números del eje Y solo en las gráficas de la izquierda.
        ax.tick_params(
            axis="y",
            left=is_left_column,
            labelleft=is_left_column,
            labelsize=TICK_LABEL_FONTSIZE,
        )

        # Label del eje X solo en las gráficas inferiores.
        if is_bottom_row:
            ax.set_xlabel(r"Volume (mm$^3$)", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_xlabel("")

        ax.tick_params(
            axis="x",
            labelsize=TICK_LABEL_FONTSIZE,
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(labels_display),
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    save_figure(
        fig,
        output_dir / f"histograms_all_variables_{task_suffix}.png",
    )



def benjamini_hochberg_correction(p_values):
    """
    Corrección FDR Benjamini-Hochberg.

    Devuelve un array con los p-values ajustados manteniendo NaN cuando los haya.
    Se implementa aquí para no depender de statsmodels.
    """
    p_values = np.asarray(p_values, dtype=float)
    p_adjusted = np.full_like(p_values, np.nan, dtype=float)

    valid_mask = np.isfinite(p_values)
    p_valid = p_values[valid_mask]

    n_tests = len(p_valid)

    if n_tests == 0:
        return p_adjusted

    order = np.argsort(p_valid)
    p_sorted = p_valid[order]
    ranks = np.arange(1, n_tests + 1)

    adjusted_sorted = p_sorted * n_tests / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    adjusted_valid = np.empty(n_tests, dtype=float)
    adjusted_valid[order] = adjusted_sorted
    p_adjusted[valid_mask] = adjusted_valid

    return p_adjusted


def format_p_value(p_value):
    """Formato simple para tablas del report."""
    if pd.isna(p_value):
        return "NA"

    if p_value < 0.001:
        return "<0.001"

    return f"{p_value:.3f}"


def format_mean_std(mean_value, std_value):
    """Formato mean ± std para tablas del report."""
    if pd.isna(mean_value) or pd.isna(std_value):
        return "NA"

    return f"{mean_value:.3f} ± {std_value:.3f}"


def compute_welch_degrees_of_freedom(var_0, var_1, n_0, n_1):
    """Calcula los grados de libertad aproximados del Welch t-test."""
    if n_0 <= 1 or n_1 <= 1:
        return np.nan

    term_0 = var_0 / n_0
    term_1 = var_1 / n_1

    numerator = (term_0 + term_1) ** 2
    denominator = (term_0 ** 2) / (n_0 - 1) + (term_1 ** 2) / (n_1 - 1)

    if denominator == 0:
        return np.nan

    return numerator / denominator


def compute_effect_sizes(mean_0, mean_1, std_0, std_1, n_0, n_1):
    """
    Calcula Cohen's d y Hedges' g para grupos independientes.

    La dirección del efecto es siempre grupo 1 - grupo 0:
        - AD - CU
        - old_AD - young_CU
    """
    if n_0 <= 1 or n_1 <= 1:
        return np.nan, np.nan

    pooled_var = (
        ((n_0 - 1) * (std_0 ** 2) + (n_1 - 1) * (std_1 ** 2))
        /
        (n_0 + n_1 - 2)
    )

    if pooled_var <= 0 or pd.isna(pooled_var):
        return np.nan, np.nan

    pooled_sd = np.sqrt(pooled_var)
    cohen_d = (mean_1 - mean_0) / pooled_sd

    # Small sample correction.
    correction = 1 - (3 / (4 * (n_0 + n_1) - 9))
    hedges_g = cohen_d * correction

    return cohen_d, hedges_g


def run_feature_t_tests(
    X,
    y,
    labels_display,
    problem_name,
    output_dir,
):
    """
    Ejecuta un t-test independiente para cada feature entre los dos grupos.

    Se usa Welch's t-test (equal_var=False), que es la opción más segura cuando
    los tamaños de grupo o las varianzas pueden ser diferentes.

    Importante:
        - El análisis se realiza con el mismo X e y que entran al modelo.
        - Por tanto, usa los mismos sujetos después del filtrado de NaN y
          WholeBrain <= 0.
        - La diferencia de medias y los effect sizes se calculan como:
          grupo 1 - grupo 0.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_suffix = make_task_suffix(problem_name)

    y_array = np.asarray(y, dtype=int)

    ttest_rows = []
    descriptives_rows = []

    for feature in X.columns:
        values = pd.to_numeric(X[feature], errors="coerce")

        vals_0 = values[y_array == 0].dropna().astype(float)
        vals_1 = values[y_array == 1].dropna().astype(float)

        n_0 = int(len(vals_0))
        n_1 = int(len(vals_1))

        mean_0 = vals_0.mean()
        mean_1 = vals_1.mean()
        std_0 = vals_0.std(ddof=1)
        std_1 = vals_1.std(ddof=1)
        var_0 = std_0 ** 2
        var_1 = std_1 ** 2

        for class_numeric, class_label, vals in [
            (0, labels_display[0], vals_0),
            (1, labels_display[1], vals_1),
        ]:
            descriptives_rows.append(
                {
                    "problem": problem_name,
                    "feature": feature,
                    "class_numeric": class_numeric,
                    "class_label": class_label,
                    "n": int(len(vals)),
                    "mean": vals.mean(),
                    "std": vals.std(ddof=1),
                    "median": vals.median(),
                    "q1": vals.quantile(0.25),
                    "q3": vals.quantile(0.75),
                    "min": vals.min(),
                    "max": vals.max(),
                }
            )

        if n_0 > 1 and n_1 > 1:
            t_statistic, p_value = stats.ttest_ind(
                vals_0,
                vals_1,
                equal_var=False,
                nan_policy="omit",
            )

            welch_df = compute_welch_degrees_of_freedom(
                var_0=var_0,
                var_1=var_1,
                n_0=n_0,
                n_1=n_1,
            )

            standard_error = np.sqrt((var_0 / n_0) + (var_1 / n_1))
            mean_difference = mean_1 - mean_0

            if standard_error > 0 and np.isfinite(welch_df):
                t_critical = stats.t.ppf(0.975, df=welch_df)
                ci95_lower = mean_difference - t_critical * standard_error
                ci95_upper = mean_difference + t_critical * standard_error
            else:
                ci95_lower = np.nan
                ci95_upper = np.nan

            cohen_d, hedges_g = compute_effect_sizes(
                mean_0=mean_0,
                mean_1=mean_1,
                std_0=std_0,
                std_1=std_1,
                n_0=n_0,
                n_1=n_1,
            )

        else:
            t_statistic = np.nan
            p_value = np.nan
            welch_df = np.nan
            mean_difference = mean_1 - mean_0
            standard_error = np.nan
            ci95_lower = np.nan
            ci95_upper = np.nan
            cohen_d = np.nan
            hedges_g = np.nan

        ttest_rows.append(
            {
                "problem": problem_name,
                "feature": feature,
                "test": "Welch independent two-sample t-test",
                "equal_variance_assumed": False,
                "group_0_label": labels_display[0],
                "group_1_label": labels_display[1],
                "n_group_0": n_0,
                "n_group_1": n_1,
                "mean_group_0": mean_0,
                "std_group_0": std_0,
                "median_group_0": vals_0.median(),
                "q1_group_0": vals_0.quantile(0.25),
                "q3_group_0": vals_0.quantile(0.75),
                "mean_group_1": mean_1,
                "std_group_1": std_1,
                "median_group_1": vals_1.median(),
                "q1_group_1": vals_1.quantile(0.25),
                "q3_group_1": vals_1.quantile(0.75),
                "mean_difference_group1_minus_group0": mean_difference,
                "standard_error_difference": standard_error,
                "ci95_lower_difference": ci95_lower,
                "ci95_upper_difference": ci95_upper,
                "t_statistic": t_statistic,
                "degrees_of_freedom_welch": welch_df,
                "p_value": p_value,
                "cohen_d_group1_minus_group0": cohen_d,
                "hedges_g_group1_minus_group0": hedges_g,
            }
        )

    t_tests = pd.DataFrame(ttest_rows)
    descriptives = pd.DataFrame(descriptives_rows)

    t_tests["p_value_fdr_bh"] = benjamini_hochberg_correction(
        t_tests["p_value"].to_numpy(dtype=float)
    )

    t_tests["significant_p_0_05"] = t_tests["p_value"] < 0.05
    t_tests["significant_fdr_0_05"] = t_tests["p_value_fdr_bh"] < 0.05

    t_tests_report = t_tests.copy()
    t_tests_report[f"{labels_display[0]}_mean_std"] = [
        format_mean_std(mean_value, std_value)
        for mean_value, std_value in zip(
            t_tests_report["mean_group_0"],
            t_tests_report["std_group_0"],
        )
    ]
    t_tests_report[f"{labels_display[1]}_mean_std"] = [
        format_mean_std(mean_value, std_value)
        for mean_value, std_value in zip(
            t_tests_report["mean_group_1"],
            t_tests_report["std_group_1"],
        )
    ]
    t_tests_report["t_df"] = [
        f"{t_value:.3f} ({df_value:.1f})"
        if pd.notna(t_value) and pd.notna(df_value)
        else "NA"
        for t_value, df_value in zip(
            t_tests_report["t_statistic"],
            t_tests_report["degrees_of_freedom_welch"],
        )
    ]
    t_tests_report["p_value_formatted"] = [
        format_p_value(p_value)
        for p_value in t_tests_report["p_value"]
    ]
    t_tests_report["p_value_fdr_bh_formatted"] = [
        format_p_value(p_value)
        for p_value in t_tests_report["p_value_fdr_bh"]
    ]

    report_cols = [
        "problem",
        "feature",
        "group_0_label",
        "group_1_label",
        "n_group_0",
        "n_group_1",
        f"{labels_display[0]}_mean_std",
        f"{labels_display[1]}_mean_std",
        "mean_difference_group1_minus_group0",
        "ci95_lower_difference",
        "ci95_upper_difference",
        "t_df",
        "p_value_formatted",
        "p_value_fdr_bh_formatted",
        "cohen_d_group1_minus_group0",
        "hedges_g_group1_minus_group0",
        "significant_p_0_05",
        "significant_fdr_0_05",
    ]

    t_tests_report = t_tests_report[report_cols]

    t_tests.to_csv(
        output_dir / f"t_test_features_{task_suffix}.csv",
        index=False,
    )

    descriptives.to_csv(
        output_dir / f"descriptive_statistics_features_{task_suffix}.csv",
        index=False,
    )

    t_tests_report.to_csv(
        output_dir / f"t_test_features_for_report_{task_suffix}.csv",
        index=False,
    )

    print("\nT-test por feature guardado:")
    print(t_tests[[
        "problem",
        "feature",
        "group_0_label",
        "group_1_label",
        "mean_difference_group1_minus_group0",
        "t_statistic",
        "degrees_of_freedom_welch",
        "p_value",
        "p_value_fdr_bh",
        "cohen_d_group1_minus_group0",
    ]])

    return t_tests, descriptives, t_tests_report


def plot_roc_curve_cv(
    y_true,
    scores,
    auc_value,
    problem_name,
    output_dir,
):
    """
    Curva ROC-AUC con predicciones out-of-fold.
    """
    task_suffix = make_task_suffix(problem_name)

    fpr, tpr, _ = roc_curve(
        y_true,
        scores,
    )

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc_value:.3f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel("False Positive Rate", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("True Positive Rate", fontsize=AXIS_LABEL_FONTSIZE)

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABEL_FONTSIZE,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE - 2)

    plt.tight_layout()

    save_figure(
        fig,
        output_dir / f"roc_auc_5fold_cv_{task_suffix}.png",
    )


def prepare_dataset_for_problem(
    df,
    problem_name,
    problem_dir,
    group_col,
    label_map,
    require_age=False,
):
    """
    Filtra NaN y valores problemáticos y devuelve X, y y metadata.

    Para CU vs AD:
        - require_age=False.
        - No se elimina a nadie por no tener AGE.
        - Solo se eliminan sujetos con NaN en group_col o en las variables.

    Para young_CU vs old_AD:
        - require_age=True.
        - Sí se exige AGE.
        - También se eliminan sujetos con NaN en las variables.
    """
    task_suffix = make_task_suffix(problem_name)

    df_model = df.copy()

    required_cols = [group_col] + FEATURES

    if require_age:
        required_cols = required_cols + [AGE_COL]

    n_before_dropna = len(df_model)

    df_model = df_model.dropna(
        subset=required_cols,
    ).copy()

    n_after_dropna = len(df_model)

    print(
        f"\nFiltrado NaN en {problem_name}: "
        f"{n_before_dropna} -> {n_after_dropna} sujetos"
    )
    print(f"Columnas usadas para filtrar NaN: {required_cols}")

    df_model = df_model[
        df_model["WholeBrain"] > 0
    ].copy()

    y = df_model[group_col].map(label_map)

    if y.isna().sum() > 0:
        raise ValueError(
            f"Hay etiquetas sin mapear en {problem_name}."
        )

    y = y.astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            f"El problema {problem_name} no contiene exactamente las clases 0 y 1."
        )

    class_counts = y.value_counts().sort_index()

    if class_counts.min() < N_SPLITS:
        raise ValueError(
            f"No hay suficientes sujetos por clase para {N_SPLITS}-fold CV en {problem_name}."
        )

    X = df_model[FEATURES].copy()

    meta_cols = get_metadata_columns(df_model)
    meta_df = df_model[meta_cols].copy()

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    meta_df = meta_df.reset_index(drop=True)

    dataset_used = meta_df.copy()
    dataset_used["y"] = y.values
    dataset_used["y_label"] = y.map(
        {v: k for k, v in label_map.items()}
    ).values

    for col in X.columns:
        dataset_used[col] = X[col].values

    dataset_used.to_csv(
        problem_dir / f"dataset_used_for_model_{task_suffix}.csv",
        index=False,
    )

    class_distribution = pd.DataFrame(
        {
            "class_numeric": class_counts.index,
            "class_label": [
                {v: k for k, v in label_map.items()}[idx]
                for idx in class_counts.index
            ],
            "n": class_counts.values,
        }
    )

    class_distribution.to_csv(
        problem_dir / f"class_distribution_{task_suffix}.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print(problem_name)
    print("=" * 80)

    print("\nSujetos usados después de filtrar NaN y WholeBrain <= 0:")
    print(class_distribution)

    return X, y, meta_df


def run_svm_problem_cv(
    X,
    y,
    meta_df,
    problem_name,
    labels_display,
    output_dir,
):
    """
    Entrena y evalúa SVM lineal con Stratified 5-fold cross-validation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_suffix = make_task_suffix(problem_name)

    labels_numeric = [0, 1]

    inverse_label_map = {
        0: labels_display[0],
        1: labels_display[1],
    }

    print("\nVariables usadas:")
    print(X.columns.tolist())

    print("\nDimensiones de X:", X.shape)

    print("\nDistribución de clases:")
    print(y.value_counts().sort_index())

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    metrics_rows = []
    split_summary_rows = []

    # Arrays para guardar predicciones out-of-fold respetando el orden original.
    y_pred_oof = np.full(shape=len(y), fill_value=-1, dtype=int)
    scores_oof = np.full(shape=len(y), fill_value=np.nan, dtype=float)
    fold_oof = np.full(shape=len(y), fill_value=-1, dtype=int)

    for fold_idx, (train_idx, valid_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):
        print(f"\nFold {fold_idx}/{N_SPLITS}")

        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        split_summary_rows.extend(
            [
                {
                    "problem": problem_name,
                    "fold": fold_idx,
                    "split": "train",
                    "class_numeric": 0,
                    "class_label": labels_display[0],
                    "n": int((y_train == 0).sum()),
                },
                {
                    "problem": problem_name,
                    "fold": fold_idx,
                    "split": "train",
                    "class_numeric": 1,
                    "class_label": labels_display[1],
                    "n": int((y_train == 1).sum()),
                },
                {
                    "problem": problem_name,
                    "fold": fold_idx,
                    "split": "validation",
                    "class_numeric": 0,
                    "class_label": labels_display[0],
                    "n": int((y_valid == 0).sum()),
                },
                {
                    "problem": problem_name,
                    "fold": fold_idx,
                    "split": "validation",
                    "class_numeric": 1,
                    "class_label": labels_display[1],
                    "n": int((y_valid == 1).sum()),
                },
            ]
        )

        model = make_model()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)

        scores_train = model.decision_function(X_train)
        scores_valid = model.decision_function(X_valid)

        metrics_rows.append(
            make_metrics_row(
                y_true=y_train,
                y_pred=y_train_pred,
                scores=scores_train,
                problem_name=problem_name,
                fold=fold_idx,
                split_name="train",
            )
        )

        metrics_rows.append(
            make_metrics_row(
                y_true=y_valid,
                y_pred=y_valid_pred,
                scores=scores_valid,
                problem_name=problem_name,
                fold=fold_idx,
                split_name="validation",
            )
        )

        y_pred_oof[valid_idx] = y_valid_pred
        scores_oof[valid_idx] = scores_valid
        fold_oof[valid_idx] = fold_idx

    if (y_pred_oof == -1).any() or np.isnan(scores_oof).any():
        raise RuntimeError(
            "Hay sujetos sin predicción out-of-fold. Revisa la cross-validation."
        )

    split_summary = pd.DataFrame(split_summary_rows)

    split_summary.to_csv(
        output_dir / f"cv_split_summary_{task_suffix}.csv",
        index=False,
    )

    print("\nDistribución por fold:")
    print(split_summary)

    metrics_by_fold = pd.DataFrame(metrics_rows)

    metrics_by_fold.to_csv(
        output_dir / f"cv_metrics_by_fold_{task_suffix}.csv",
        index=False,
    )

    metrics_summary = summarize_cv_metrics(metrics_by_fold)

    metrics_summary.to_csv(
        output_dir / f"cv_metrics_summary_{task_suffix}.csv",
        index=False,
    )

    print("\nMétricas por fold guardadas:")
    print(metrics_by_fold)

    print("\nResumen CV mean/std guardado:")
    print(metrics_summary)

    # Métricas globales con todas las predicciones out-of-fold.
    oof_metrics = pd.DataFrame(
        [
            make_metrics_row(
                y_true=y,
                y_pred=y_pred_oof,
                scores=scores_oof,
                problem_name=problem_name,
                fold="all_oof",
                split_name="out_of_fold",
            )
        ]
    )

    oof_metrics.to_csv(
        output_dir / f"cv_metrics_out_of_fold_global_{task_suffix}.csv",
        index=False,
    )

    cm_cv = confusion_matrix(
        y,
        y_pred_oof,
        labels=labels_numeric,
    )

    pd.DataFrame(
        cm_cv,
        index=[
            f"True_{labels_display[0]}",
            f"True_{labels_display[1]}",
        ],
        columns=[
            f"Pred_{labels_display[0]}",
            f"Pred_{labels_display[1]}",
        ],
    ).to_csv(output_dir / f"confusion_matrix_5fold_cv_{task_suffix}.csv")

    classification_reports = make_classification_report_df(
        y_true=y,
        y_pred=y_pred_oof,
        labels_numeric=labels_numeric,
        labels_display=labels_display,
        problem_name=problem_name,
    )

    classification_reports.to_csv(
        output_dir / f"classification_report_5fold_cv_{task_suffix}.csv",
        index=False,
    )

    report_text = classification_report(
        y,
        y_pred_oof,
        labels=labels_numeric,
        target_names=labels_display,
        zero_division=0,
    )

    with open(
        output_dir / f"classification_report_5fold_cv_{task_suffix}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report_text)

    predictions = meta_df.copy()
    predictions["fold"] = fold_oof
    predictions["y_true"] = y.values
    predictions["y_true_label"] = [
        inverse_label_map[int(v)] for v in y.values
    ]
    predictions["y_pred"] = y_pred_oof
    predictions["y_pred_label"] = [
        inverse_label_map[int(v)] for v in y_pred_oof
    ]
    predictions["decision_function_score"] = scores_oof
    predictions["correct"] = predictions["y_true"] == predictions["y_pred"]

    predictions.to_csv(
        output_dir / f"predictions_5fold_cv_{task_suffix}.csv",
        index=False,
    )

    plot_confusion_matrix_cv(
        cm_cv=cm_cv,
        labels_display=labels_display,
        problem_name=problem_name,
        output_dir=output_dir,
    )

    plot_histograms_all_variables(
        X=X,
        y=y,
        labels_display=labels_display,
        problem_name=problem_name,
        output_dir=output_dir,
    )

    auc_oof = roc_auc_score(y, scores_oof)

    plot_roc_curve_cv(
        y_true=y,
        scores=scores_oof,
        auc_value=auc_oof,
        problem_name=problem_name,
        output_dir=output_dir,
    )

    return metrics_by_fold, metrics_summary, classification_reports, oof_metrics


# =============================================================================
# 4. Cargar datos
# =============================================================================
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

validate_columns(
    df,
    [TARGET_COL, AGE_COL] + FEATURES,
)

df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()

for col in [AGE_COL] + FEATURES:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

print("\nDimensiones iniciales:", df.shape)

print("\nDistribución inicial en DX_binary:")
print(df[TARGET_COL].value_counts(dropna=False))

print("\nResumen de AGE:")
print(df[AGE_COL].describe())


# =============================================================================
# 5. Threshold de edad
# =============================================================================
age_mean = df[AGE_COL].dropna().mean()

df_age_threshold = pd.DataFrame(
    {
        "threshold_type": ["mean"],
        "threshold_value": [age_mean],
        "n_subjects_used_to_compute_threshold": [
            int(df[AGE_COL].notna().sum())
        ],
    }
)

df_age_threshold.to_csv(
    RESULTS_DIR / "age_threshold_mean_all_subjects.csv",
    index=False,
)

print("\nThreshold de edad:")
print(df_age_threshold)


# =============================================================================
# 6. Problema 1: CU vs AD
# =============================================================================
problem_1_name = "CU vs AD"
problem_1_dir = RESULTS_DIR / problem_1_name
problem_1_dir.mkdir(parents=True, exist_ok=True)

problem_1_suffix = make_task_suffix(problem_1_name)

df_cu_ad = df[df[TARGET_COL].isin(["CU", "AD"])].copy()
df_cu_ad["group_age_dx"] = df_cu_ad[TARGET_COL]

df_cu_ad.to_csv(
    problem_1_dir / f"subjects_before_mri_filter_{problem_1_suffix}.csv",
    index=False,
)

label_map_cu_ad = {
    "CU": 0,
    "AD": 1,
}

X_1, y_1, meta_1 = prepare_dataset_for_problem(
    df=df_cu_ad,
    problem_name=problem_1_name,
    problem_dir=problem_1_dir,
    group_col="group_age_dx",
    label_map=label_map_cu_ad,
    require_age=False,
)

t_tests_1, descriptives_1, t_tests_report_1 = run_feature_t_tests(
    X=X_1,
    y=y_1,
    labels_display=["CU", "AD"],
    problem_name=problem_1_name,
    output_dir=problem_1_dir,
)

metrics_by_fold_1, metrics_summary_1, reports_1, oof_metrics_1 = run_svm_problem_cv(
    X=X_1,
    y=y_1,
    meta_df=meta_1,
    problem_name=problem_1_name,
    labels_display=["CU", "AD"],
    output_dir=problem_1_dir,
)


# =============================================================================
# 7. Problema 2: young_CU vs old_AD
# =============================================================================
problem_2_name = "young CU vs old AD"
problem_2_dir = RESULTS_DIR / problem_2_name
problem_2_dir.mkdir(parents=True, exist_ok=True)

problem_2_suffix = make_task_suffix(problem_2_name)

mask_young_CU = (
    (df[TARGET_COL] == "CU")
    &
    (df[AGE_COL] <= age_mean)
)

mask_old_ad = (
    (df[TARGET_COL] == "AD")
    &
    (df[AGE_COL] >= age_mean)
)

df_young_old = df[mask_young_CU | mask_old_ad].copy()

df_young_old["age_threshold_value"] = age_mean

df_young_old["group_age_dx"] = np.where(
    df_young_old[TARGET_COL] == "CU",
    "young_CU",
    "old_AD",
)

df_young_old.to_csv(
    problem_2_dir / f"subjects_before_mri_filter_{problem_2_suffix}.csv",
    index=False,
)

label_map_young_old = {
    "young_CU": 0,
    "old_AD": 1,
}

X_2, y_2, meta_2 = prepare_dataset_for_problem(
    df=df_young_old,
    problem_name=problem_2_name,
    problem_dir=problem_2_dir,
    group_col="group_age_dx",
    label_map=label_map_young_old,
    require_age=True,
)

t_tests_2, descriptives_2, t_tests_report_2 = run_feature_t_tests(
    X=X_2,
    y=y_2,
    labels_display=["young_CU", "old_AD"],
    problem_name=problem_2_name,
    output_dir=problem_2_dir,
)

metrics_by_fold_2, metrics_summary_2, reports_2, oof_metrics_2 = run_svm_problem_cv(
    X=X_2,
    y=y_2,
    meta_df=meta_2,
    problem_name=problem_2_name,
    labels_display=["young_CU", "old_AD"],
    output_dir=problem_2_dir,
)


# =============================================================================
# 8. Tablas finales combinadas
# =============================================================================
all_metrics_by_fold = pd.concat(
    [metrics_by_fold_1, metrics_by_fold_2],
    axis=0,
    ignore_index=True,
)

all_metrics_summary = pd.concat(
    [metrics_summary_1, metrics_summary_2],
    axis=0,
    ignore_index=True,
)

all_oof_metrics = pd.concat(
    [oof_metrics_1, oof_metrics_2],
    axis=0,
    ignore_index=True,
)

all_reports = pd.concat(
    [reports_1, reports_2],
    axis=0,
    ignore_index=True,
)

all_t_tests = pd.concat(
    [t_tests_1, t_tests_2],
    axis=0,
    ignore_index=True,
)

all_descriptives = pd.concat(
    [descriptives_1, descriptives_2],
    axis=0,
    ignore_index=True,
)

all_t_tests_report = pd.concat(
    [t_tests_report_1, t_tests_report_2],
    axis=0,
    ignore_index=True,
)

all_metrics_by_fold.to_csv(
    RESULTS_DIR / "cv_metrics_by_fold_all_problems.csv",
    index=False,
)

all_metrics_summary.to_csv(
    RESULTS_DIR / "cv_metrics_summary_all_problems.csv",
    index=False,
)

all_oof_metrics.to_csv(
    RESULTS_DIR / "cv_metrics_out_of_fold_global_all_problems.csv",
    index=False,
)

all_reports.to_csv(
    RESULTS_DIR / "classification_report_5fold_cv_all_problems.csv",
    index=False,
)

all_t_tests.to_csv(
    RESULTS_DIR / "t_test_features_all_problems.csv",
    index=False,
)

all_descriptives.to_csv(
    RESULTS_DIR / "descriptive_statistics_features_all_problems.csv",
    index=False,
)

all_t_tests_report.to_csv(
    RESULTS_DIR / "t_test_features_for_report_all_problems.csv",
    index=False,
)

print("\n" + "=" * 80)
print("PROCESO TERMINADO")
print("=" * 80)

print("\nTabla final de métricas por fold:")
print(all_metrics_by_fold)

print("\nTabla final de mean/std de cross-validation:")
print(all_metrics_summary)

print("\nMétricas globales con predicciones out-of-fold:")
print(all_oof_metrics)

print("\nTabla final de t-test por feature:")
print(all_t_tests[[
    "problem",
    "feature",
    "mean_difference_group1_minus_group0",
    "t_statistic",
    "degrees_of_freedom_welch",
    "p_value",
    "p_value_fdr_bh",
    "cohen_d_group1_minus_group0",
]])

print("\nResultados guardados en:")
print(RESULTS_DIR)
