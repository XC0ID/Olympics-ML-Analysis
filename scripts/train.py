import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data.loader           import load_all
from src.data.cleaner          import clean_medals, clean_countries
from src.data.validator        import validate_medals, validate_countries
from src.features.builder      import build_features, get_feature_cols
from src.models.regression     import build_rf_regressor, cross_validate_reg
from src.models.classification import build_dominance_labels, build_classifier
from src.models.clustering     import build_kmeans, find_optimal_k
from src.evaluation.metrics    import regression_metrics, classification_metrics
from src.evaluation.plotter    import (plot_medal_trends, plot_feature_importance,
                                        plot_actual_vs_pred, plot_cluster_heatmap,
                                        plot_elbow)
from src.utils.helpers         import save_json, save_csv

BASE        = Path(__file__).parent.parent
MODEL_DIR   = BASE / "models" / "trained"
METRICS_DIR = BASE / "results" / "metrics"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("\n=== 1. Loading data ===")
    medals_raw, countries_raw = load_all(config_path=str(BASE / "config/config.yaml"))
    validate_medals(medals_raw)
    validate_countries(countries_raw)
    medals    = clean_medals(medals_raw)
    countries = clean_countries(countries_raw)

    print("\n=== 2. Building features ===")
    features  = build_features(medals, countries)
    features  = features.dropna()
    feat_cols = get_feature_cols(features)
    print(f"Features shape: {features.shape}")
    print(f"Feature columns: {feat_cols}")
    features.to_csv(BASE / "data/processed/features.csv", index=False)
    plot_medal_trends(features)

    X = features[feat_cols].values
    y = features["TotalMedals"].values

    # ── Regression ──────────────────────────────────────────
    print("\n=== 3. Regression ===")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)
    rf   = build_rf_regressor()
    rf.fit(X_tr, y_tr)
    mets = regression_metrics(y_te, rf.predict(X_te))
    mets.update(cross_validate_reg(rf, X, y))
    print(f"  RMSE: {mets['rmse']}  R2: {mets['r2']}")
    plot_feature_importance(rf, feat_cols, "regression_feature_importance")
    plot_actual_vs_pred(y_te, rf.predict(X_te))
    joblib.dump(rf, MODEL_DIR / "regression_rf.pkl")
    print(f"  Saved: {MODEL_DIR / 'regression_rf.pkl'}")

    # ── Classification ──────────────────────────────────────
    print("\n=== 4. Classification ===")
    dom      = build_dominance_labels(medals)
    feat_cls = features.merge(dom, on=["Country", "Year"], how="inner").dropna()
    clf_mets = {}
    if len(feat_cls) > 50:
        fc   = get_feature_cols(feat_cls)
        Xc   = feat_cls[fc].values
        le   = LabelEncoder()
        yc   = le.fit_transform(feat_cls["TopSport"])
        counts   = np.bincount(yc)
        stratify = yc if counts.min() > 1 else None
        Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
            Xc, yc, test_size=0.2, random_state=42, stratify=stratify)
        clf = build_classifier()
        clf.fit(Xc_tr, yc_tr)
        clf_mets = {"accuracy": classification_metrics(yc_te, clf.predict(Xc_te))["accuracy"]}
        print(f"  Accuracy: {clf_mets['accuracy']}")
        joblib.dump(clf, MODEL_DIR / "classification_rf.pkl")
        joblib.dump(le,  MODEL_DIR / "label_encoder.pkl")
        print(f"  Saved: classification_rf.pkl + label_encoder.pkl")
    else:
        print("  Skipping — not enough samples.")

    # ── Clustering ──────────────────────────────────────────
    print("\n=== 5. Clustering ===")
    country_agg = features.groupby("Country")[feat_cols].mean().dropna().reset_index()
    Xk          = country_agg[feat_cols].values
    k_scores    = find_optimal_k(Xk)
    best_k      = max(k_scores, key=lambda k: k_scores[k]["silhouette"])
    print(f"  Best k={best_k}  silhouette={k_scores[best_k]['silhouette']}")
    plot_elbow(k_scores)
    km = build_kmeans(n_clusters=best_k)
    km.fit(Xk)
    country_agg["Cluster"] = km.named_steps["kmeans"].labels_
    plot_cluster_heatmap(country_agg, feat_cols)
    save_csv(country_agg[["Country", "Cluster"]], str(METRICS_DIR / "country_clusters.csv"))
    joblib.dump(km, MODEL_DIR / "clustering_kmeans.pkl")
    print(f"  Saved: clustering_kmeans.pkl")

    # ── Save all metrics ────────────────────────────────────
    save_json({"regression": mets, "classification": clf_mets,
               "clustering_k_scores": k_scores},
              str(METRICS_DIR / "metrics.json"))

    print("\n=== Done! Check models/trained/ and results/ ===")

if __name__ == "__main__":
    main()