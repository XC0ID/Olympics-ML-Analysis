import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.cleaner     import clean_medals, clean_countries
from src.features.builder import build_features, get_feature_cols

def test_has_required_cols(sample_medals, sample_countries):
    df = build_features(clean_medals(sample_medals), clean_countries(sample_countries))
    assert "TotalMedals" in df.columns
    assert "WeightedScore" in df.columns

def test_feature_cols_are_numeric(sample_medals, sample_countries):
    import pandas as pd
    df   = build_features(clean_medals(sample_medals), clean_countries(sample_countries))
    cols = get_feature_cols(df)
    for c in cols:
        assert pd.api.types.is_numeric_dtype(df[c])