import pandas as pd
from sklearn.preprocessing import StandardScaler

def encode_season(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IsSummer"] = (df["Season"] == "Summer").astype(int)
    return df

def fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    return df

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler