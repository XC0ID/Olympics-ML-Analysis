import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, accuracy_score,
                              classification_report, silhouette_score)

def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
    }

def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "report":   classification_report(y_true, y_pred, output_dict=True),
    }

def clustering_metrics(X_scaled, labels) -> dict:
    return {"silhouette": round(float(silhouette_score(X_scaled, labels)), 4)}