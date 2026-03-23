import pandas as pd

MEDAL_WEIGHTS = {"Gold": 3, "Silver": 2, "Bronze": 1}

def build_features(medals: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    df = medals.copy()
    df["MedalScore"] = df["Medal"].map(MEDAL_WEIGHTS)

    agg = (df.groupby(["Country", "Year", "Season"])
             .agg(
                 TotalMedals   = ("Medal",        "count"),
                 GoldMedals    = ("Medal",        lambda x: (x == "Gold").sum()),
                 SilverMedals  = ("Medal",        lambda x: (x == "Silver").sum()),
                 BronzeMedals  = ("Medal",        lambda x: (x == "Bronze").sum()),
                 WeightedScore = ("MedalScore",   "sum"),
             ).reset_index())

    agg = agg.sort_values(["Country", "Season", "Year"])
    agg["PrevMedals"]   = (agg.groupby(["Country", "Season"])["TotalMedals"]
                              .shift(1).fillna(0))
    agg["RollingMean3"] = (agg.groupby(["Country", "Season"])["TotalMedals"]
                              .transform(lambda x: x.shift(1)
                              .rolling(3, min_periods=1).mean()).fillna(0))
    agg["MedalDelta"]   = agg["TotalMedals"] - agg["PrevMedals"]
    agg["IsSummer"]     = (agg["Season"] == "Summer").astype(int)

    if not countries.empty:
        agg = agg.merge(countries, on="Country", how="left")

    return agg

def get_feature_cols(df: pd.DataFrame):
    exclude = {"Country", "Year", "Season", "TotalMedals",
               "GoldMedals", "SilverMedals", "BronzeMedals", "WeightedScore"}
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]