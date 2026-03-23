import pandas as pd

def clean_medals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Country", "Year", "Medal"])
    df["Year"]  = pd.to_numeric(df["Year"], errors="coerce")
    df["Medal"] = df["Medal"].str.strip().str.title()
    df = df[df["Medal"].isin(["Gold", "Silver", "Bronze"])]
    df = df.drop_duplicates()
    return df.reset_index(drop=True)

def clean_countries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Country"])
    df["Country"] = df["Country"].str.strip()
    for col in df.select_dtypes(include="object").columns:
        if col != "Country":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)