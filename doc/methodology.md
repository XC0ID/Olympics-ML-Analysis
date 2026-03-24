# Methodology

## 1. Data loading
- SummerSD.csv and WinterSD.csv are loaded and merged
- Season column is added (Summer / Winter)
- CountriesSD.csv is merged on Country column

## 2. Data cleaning
- Null rows removed from Country, Year, Medal columns
- Invalid medal values removed (only Gold, Silver, Bronze kept)
- Duplicate rows removed
- Country names stripped of whitespace

## 3. Feature engineering
- Medal counts aggregated per country per year per season
- Weighted score: Gold=3, Silver=2, Bronze=1
- Lag feature: previous games medal count
- Rolling mean over last 3 games
- Medal delta: improvement over last games
- Country socioeconomic data merged (Population, GDP per Capita)
- Nulls filled with column median

## 4. Models

### Regression — Random Forest
- Target: TotalMedals
- Algorithm: Random Forest Regressor
- Evaluation: RMSE, MAE, R2, 5-fold cross validation

### Classification — Random Forest
- Target: TopSport (dominant sport per country per year)
- Algorithm: Random Forest Classifier
- Evaluation: Accuracy, Classification report

### Clustering — KMeans
- Target: Country performance groups
- Algorithm: KMeans with optimal k selection
- Evaluation: Silhouette score, Elbow method

## 5. Evaluation
- Train/test split: 80% train, 20% test
- Cross validation: 5 folds
- All metrics saved to results/metrics/metrics.json