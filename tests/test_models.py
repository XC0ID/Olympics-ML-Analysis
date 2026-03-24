import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.regression import build_rf_regressor
from src.models.clustering import build_kmeans

def test_rf_fits_and_predicts():
    X = np.random.rand(50, 4)
    y = np.random.rand(50) * 10
    model = build_rf_regressor(n_estimators=10)
    model.fit(X, y)
    assert len(model.predict(X)) == 50

def test_kmeans_pipeline():
    X  = np.random.rand(30, 3)
    km = build_kmeans(n_clusters=3)
    km.fit(X)
    assert hasattr(km.named_steps["kmeans"], "labels_")