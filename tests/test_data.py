import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.cleaner import clean_medals, clean_countries

def test_removes_nulls(sample_medals):
    dirty = sample_medals.copy()
    dirty.loc[0, "Country"] = None
    assert clean_medals(dirty)["Country"].isna().sum() == 0

def test_valid_medals_only(sample_medals):
    dirty = sample_medals.copy()
    dirty.loc[0, "Medal"] = "Platinum"
    assert set(clean_medals(dirty)["Medal"]).issubset({"Gold","Silver","Bronze"})

def test_no_duplicates(sample_medals):
    dirty = pd.concat([sample_medals, sample_medals])
    assert clean_medals(dirty).duplicated().sum() == 0

def test_countries_strips_whitespace(sample_countries):
    dirty = sample_countries.copy()
    dirty["Country"] = "  " + dirty["Country"] + "  "
    result = clean_countries(dirty)
    assert result["Country"].str.strip().equals(result["Country"])