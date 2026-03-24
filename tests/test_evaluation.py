import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.metrics import regression_metrics, clustering_metrics
from sklearn.preprocessing  import StandardScaler

def test_regression_metric_keys():
    m = regression_metrics(np.array([1,2,3,4,5], dtype=float),
                           np.array([1.1,1.9,3.2,3.8,5.1]))
    assert {"rmse","mae","r2"}.issubset(m.keys())

def test_clustering_metric():
    X      = np.random.rand(20, 3)
    Xs     = StandardScaler().fit_transform(X)
    labels = np.array([0]*10 + [1]*10)
    assert "silhouette" in clustering_metrics(Xs, labels)