from sklearn.model_selection import KFold, cross_val_score

def kfold_cv(model, X, y, scoring="neg_root_mean_squared_error", n_splits=5):
    kf     = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    return {"mean": round(-scores.mean(), 4), "std": round(scores.std(), 4)}