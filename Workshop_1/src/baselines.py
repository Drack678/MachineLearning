"""
baselines.py
Consolidación y evaluación de modelos base (Regresión Logística y Random Forest)
para predicción de caracteres.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# -------------------------------------------------------------------------
# 1. Configuración de Entorno y Rutas
# -------------------------------------------------------------------------
# Localiza la carpeta 'src' y la raíz del proyecto (Workshop_1) de forma absoluta
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent              # .../Workshop_1/src
PROJECT_ROOT = SRC_DIR.parent              # .../Workshop_1

# Agregar tanto la raíz como src al PATH si no están
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    # Intento 1: importación como paquete (si ejecutas desde la raíz del proyecto)
    from src.dataset import load_split
    from src.config import SEED, set_seeds
    set_seeds()
except ImportError:
    try:
        # Intento 2: importación directa si estás dentro de 'src'
        from dataset import load_split
        from config import SEED, set_seeds
        set_seeds()
    except ImportError as e:
        raise ImportError(
            f"No se pudo encontrar 'dataset.py' ni 'config.py'. "
            f"Verifica que existan en '{SRC_DIR}' o '{PROJECT_ROOT}'. Detalle: {e}"
        )

# Parámetros globales
CONTEXT_LENGTH = 5
MAX_TRAIN_ROWS = 30000
MAX_VAL_ROWS = 10000
MAX_TEST_ROWS = 10000


# -------------------------------------------------------------------------
# 2. Funciones Auxiliares
# -------------------------------------------------------------------------
def context_matrix(series: pd.Series, ctx_length: int) -> np.ndarray:
    """Extrae los últimos `ctx_length` caracteres de cada texto en la serie."""
    characters = [list(str(text)[-ctx_length:]) for text in series]
    return np.asarray(characters, dtype=object)


def evaluate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas estándar de clasificación multiclase."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision (Weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (Weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (Weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def compute_macro_roc_auc_ovr(model, x_data, y_true: np.ndarray) -> tuple[float, int, int]:
    """
    Calcula el Macro AUC-ROC (One-vs-Rest) adaptado a clases presentes
    en la partición de prueba/validación.
    """
    y_proba = model.predict_proba(x_data)
    model_classes = model.classes_

    known_mask = np.isin(y_true, model_classes)
    y_true_filtered = y_true[known_mask]
    y_proba_filtered = y_proba[known_mask]

    class_to_index = {class_id: idx for idx, class_id in enumerate(model_classes)}
    y_true_index = np.array([class_to_index[cls] for cls in y_true_filtered])

    present_classes = np.unique(y_true_index)
    if len(present_classes) < 2:
        return np.nan, len(y_true_filtered), len(present_classes)

    y_true_binary = np.zeros((len(y_true_index), len(present_classes)))
    for i, cls_idx in enumerate(present_classes):
        y_true_binary[:, i] = (y_true_index == cls_idx)

    y_proba_present = y_proba_filtered[:, present_classes]
    
    # Re-normalizar probabilidades a las clases presentes si es necesario
    row_sums = y_proba_present.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_proba_present = y_proba_present / row_sums

    macro_auc = roc_auc_score(
        y_true_binary,
        y_proba_present,
        average="macro",
        multi_class="ovr"
    )
    return macro_auc, len(y_true_filtered), len(present_classes)


def plot_and_save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str):
    """Genera y guarda la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Confusion matrix guardada en: {filename}")


# -------------------------------------------------------------------------
# 3. Carga y Muestreo de Datos
# -------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Cargando particiones del dataset...")
    print("=" * 60)
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    # Muestreo estratificado/controlado por SEED
    train_sample = train_df.sample(n=min(MAX_TRAIN_ROWS, len(train_df)), random_state=SEED)
    val_sample = val_df.sample(n=min(MAX_VAL_ROWS, len(val_df)), random_state=SEED)
    test_sample = test_df.sample(n=min(MAX_TEST_ROWS, len(test_df)), random_state=SEED)

    print(f"Train sample: {len(train_sample)} filas")
    print(f"Val sample:   {len(val_sample)} filas")
    print(f"Test sample:  {len(test_sample)} filas")

    # ---------------------------------------------------------------------
    # 4. Extracción de Features (Context Window) y One-Hot Encoding
    # ---------------------------------------------------------------------
    print("\nProcesando contexto de caracteres...")
    x_train_chars = context_matrix(train_sample["x"], CONTEXT_LENGTH)
    x_val_chars = context_matrix(val_sample["x"], CONTEXT_LENGTH)
    x_test_chars = context_matrix(test_sample["x"], CONTEXT_LENGTH)

    encoder = OneHotEncoder(handle_unknown="ignore")
    x_train = encoder.fit_transform(x_train_chars)
    x_val = encoder.transform(x_val_chars)
    x_test = encoder.transform(x_test_chars)

    # Nota: se usa 'y_id' uniformemente para evitar incompatibilidad de tipos
    y_train = train_sample["y_id"].to_numpy()
    y_val = val_sample["y_id"].to_numpy()
    y_test = test_sample["y_id"].to_numpy()

    print(f"Dimensiones X_train codificado: {x_train.shape}")
    print(f"Dimensiones X_val codificado:   {x_val.shape}")
    print(f"Dimensiones X_test codificado:  {x_test.shape}")

    # =====================================================================
    # BASELINE 1: REGRESIÓN LOGÍSTICA
    # =====================================================================
    print("\n" + "=" * 60)
    print("Entrenando Baseline 1: Regresión Logística...")
    print("=" * 60)

    lr_model = LogisticRegression(
        C=1.0,
        solver="saga",
        max_iter=1000,
        random_state=SEED,
        n_jobs=-1
    )
    lr_model.fit(x_train, y_train)

    # Validación
    y_val_pred_lr = lr_model.predict(x_val)
    val_metrics_lr = evaluate_classification_metrics(y_val, y_val_pred_lr)
    val_auc_lr, _, _ = compute_macro_roc_auc_ovr(lr_model, x_val, y_val)
    val_metrics_lr["Macro AUC-ROC"] = val_auc_lr

    # Test
    y_test_pred_lr = lr_model.predict(x_test)
    test_metrics_lr = evaluate_classification_metrics(y_test, y_test_pred_lr)
    test_auc_lr, _, _ = compute_macro_roc_auc_ovr(lr_model, x_test, y_test)
    test_metrics_lr["Macro AUC-ROC"] = test_auc_lr

    print("\n--- Métricas Regresión Logística (Test) ---")
    for k, v in test_metrics_lr.items():
        print(f"{k:22s}: {v:.4f}")

    plot_and_save_confusion_matrix(
        y_val, y_val_pred_lr,
        "Logistic Regression - Validation Confusion Matrix",
        "lr_cm_validation.png"
    )
    plot_and_save_confusion_matrix(
        y_test, y_test_pred_lr,
        "Logistic Regression - Test Confusion Matrix",
        "lr_cm_test.png"
    )

    # =====================================================================
    # BASELINE 2: RANDOM FOREST (BÚSQUEDA DE HIPERPARÁMETROS)
    # =====================================================================
    print("\n" + "=" * 60)
    print("Optimizando Baseline 2: Random Forest...")
    print("=" * 60)

    n_estimators_list = [100, 200, 400]
    depth_list = [12, 20, 24]
    min_leaf_list = [2, 5]
    max_feat_list = ["sqrt", 0.5]
    class_weights_list = ["balanced", "balanced_subsample"]

    rf_experiments = []
    best_macro_f1 = -1.0
    best_rf_model = None
    best_rf_name = ""

    for n in n_estimators_list:
        for d in depth_list:
            for l in min_leaf_list:
                for f in max_feat_list:
                    for w in class_weights_list:
                        cfg_name = f"rf_{n}_{d}_{l}_{f}_{w}"
                        rf = RandomForestClassifier(
                            n_estimators=n,
                            max_depth=d,
                            min_samples_leaf=l,
                            max_features=f,
                            class_weight=w,
                            n_jobs=-1,
                            random_state=SEED
                        )
                        rf.fit(x_train, y_train)
                        val_pred_rf = rf.predict(x_val)

                        acc = accuracy_score(y_val, val_pred_rf)
                        bal_acc = balanced_accuracy_score(y_val, val_pred_rf)
                        macro_f1 = f1_score(y_val, val_pred_rf, average="macro", zero_division=0)
                        weighted_f1 = f1_score(y_val, val_pred_rf, average="weighted", zero_division=0)

                        rf_experiments.append({
                            "name": cfg_name,
                            "accuracy": acc,
                            "balanced_accuracy": bal_acc,
                            "macro_f1": macro_f1,
                            "weighted_f1": weighted_f1,
                        })

                        if macro_f1 > best_macro_f1:
                            best_macro_f1 = macro_f1
                            best_rf_model = rf
                            best_rf_name = cfg_name

                        print(f"{cfg_name} -> Macro-F1: {macro_f1:.4f} | Acc: {acc:.4f}")

    rf_results_df = pd.DataFrame(rf_experiments).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    print("\nTop 5 Configuraciones de Random Forest:")
    print(rf_results_df.head(5).to_string())

    # Evaluación completa del mejor Random Forest
    print(f"\nEvaluando el mejor modelo RF ({best_rf_name}) en Validation y Test...")
    y_val_pred_rf = best_rf_model.predict(x_val)
    val_metrics_rf = evaluate_classification_metrics(y_val, y_val_pred_rf)
    val_auc_rf, _, _ = compute_macro_roc_auc_ovr(best_rf_model, x_val, y_val)
    val_metrics_rf["Macro AUC-ROC"] = val_auc_rf

    y_test_pred_rf = best_rf_model.predict(x_test)
    test_metrics_rf = evaluate_classification_metrics(y_test, y_test_pred_rf)
    test_auc_rf, _, _ = compute_macro_roc_auc_ovr(best_rf_model, x_test, y_test)
    test_metrics_rf["Macro AUC-ROC"] = test_auc_rf

    plot_and_save_confusion_matrix(
        y_test, y_test_pred_rf,
        f"Random Forest ({best_rf_name}) - Test Confusion Matrix",
        "rf_cm_test.png"
    )

    # =====================================================================
    # 5. Resumen Comparativo Final
    # =====================================================================
    print("\n" + "=" * 60)
    print("RESUMEN COMPARATIVO FINAL DE BASELINES")
    print("=" * 60)

    summary_df = pd.DataFrame({
        "Métrica": list(test_metrics_lr.keys()),
        "LR (Val)": [val_metrics_lr[m] for m in test_metrics_lr.keys()],
        "LR (Test)": [test_metrics_lr[m] for m in test_metrics_lr.keys()],
        "Best RF (Val)": [val_metrics_rf[m] for m in test_metrics_lr.keys()],
        "Best RF (Test)": [test_metrics_rf[m] for m in test_metrics_lr.keys()],
    })
    print(summary_df.round(4).to_string(index=False))

    summary_df.to_csv("baselines_summary.csv", index=False)
    print("\nResultados consolidados guardados en 'baselines_summary.csv'.")


if __name__ == "__main__":
    main()