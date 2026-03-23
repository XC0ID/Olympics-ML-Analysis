from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def build_voting_regressor():
    return VotingRegressor([
        ("rf",    RandomForestRegressor(n_estimators=100, random_state=42)),
        ("gbr",   GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("ridge", Ridge()),
    ])

def build_stacking_regressor():
    estimators = [
        ("rf",  RandomForestRegressor(n_estimators=100, random_state=42)),
        ("gbr", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]
    return StackingRegressor(estimators=estimators, final_estimator=Ridge())