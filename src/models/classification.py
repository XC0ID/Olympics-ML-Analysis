import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def build_dominance_labels(medals: pd.DataFrame) -> pd.DataFrame:
    sport_col = "Sport" if "Sport" in medals.columns else medals.columns[3]
    counts = (medals[medals["Medal"] == "Gold"]
              .groupby(["Country", "Year", sport_col])["Medal"]
              .count().reset_index(name="SportGolds"))
    idx = counts.groupby(["Country", "Year"])["SportGolds"].idxmax()
    return counts.loc[idx, ["Country", "Year", sport_col]].rename(
        columns={sport_col: "TopSport"})

def build_classifier(n_estimators=200, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state,
        class_weight="balanced", n_jobs=-1)

def cross_validate_clf(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv)
    return {"acc_mean": round(scores.mean(), 4),
            "acc_std":  round(scores.std(), 4)}