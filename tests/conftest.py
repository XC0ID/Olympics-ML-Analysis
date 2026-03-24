import pytest
import pandas as pd

@pytest.fixture
def sample_medals():
    return pd.DataFrame({
        "Country": ["USA","USA","GBR","GBR","AUS"],
        "Year":    [2000, 2000, 2004, 2004, 2008],
        "Sport":   ["Swimming","Athletics","Rowing","Cycling","Swimming"],
        "Medal":   ["Gold","Silver","Bronze","Gold","Silver"],
        "Season":  ["Summer"]*5,
    })

@pytest.fixture
def sample_countries():
    return pd.DataFrame({
        "Country":    ["USA","GBR","AUS"],
        "Population": [330000000, 67000000, 26000000],
        "GDP":        [21000, 2800, 1400],
    })