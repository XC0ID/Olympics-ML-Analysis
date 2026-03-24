# 🏅 Olympics Summer & Winter Analysis 
---

## 📌 Overview
This project provides a complete end-to-end analysis of historical Olympic data from both Summer and Winter Games. It combines data analysis, feature engineering, and machine learning to uncover insights about country performance, medal trends, and participation patterns.

The project is designed using a **production-level machine learning architecture**, making it scalable, modular, and easy to extend for real-world applications.

---

## 🎯 Objectives
- Analyze historical Olympic datasets (Summer & Winter)
- Clean and preprocess raw data
- Perform exploratory data analysis (EDA)
- Engineer meaningful features
- Build machine learning models (regression & classification)
- Evaluate model performance
- Create a reusable ML pipeline structure

---

## 📊 Dataset
The project uses the following datasets:
- **Summer Olympics Dataset**
- **Winter Olympics Dataset**
- **Country Metadata Dataset**

### Data Includes:
- Athlete details
- Country participation
- Medal counts (Gold, Silver, Bronze)
- Event and sport categories
- Year-wise performance trends

---

## 🏗️ Complete Project Structure
```
Olympics-ML-Analysis/
│
├── README.md                          # Project overview, setup, usage
├── LICENSE                            # MIT/Apache license
├── .gitignore                         # Git ignore patterns
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── Makefile                           # Common commands
│
├── data/
│   ├── raw/                           # Original immutable data
│   │   ├── CountriesSD.csv
│   │   ├── SummerSD.csv
│   │   └── .gitkeep
│   ├── processed/                     # Cleaned transformed data
│   │   ├── countries_processed.csv
│   │   ├── summer_processed.csv
│   │   └── .gitkeep
│   ├── external/                      # External sources
│   │   └── .gitkeep
│   └── README.md                      # Data dictionary
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── README.md
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── config.py                      # Configuration
│   ├── logger.py                      # Logging setup
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Data loading
│   │   ├── cleaner.py                 # Cleaning functions
│   │   ├── preprocessor.py            # Preprocessing pipeline
│   │   └── validator.py               # Data validation
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── builder.py                 # Feature engineering
│   │   └── selector.py                # Feature selection
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                    # Base model class
│   │   ├── regression.py              # Regression models
│   │   ├── classification.py          # Classification models
│   │   ├── ensemble.py                # Ensemble methods
│   │   └── trainer.py                 # Training logic
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Evaluation metrics
│   │   ├── validator.py               # Cross-validation
│   │   └── plotter.py                 # Visualizations
│   │
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py                 # Utilities
│       └── constants.py               # Constants
│
├── models/
│   ├── trained/                       # Saved models
│   │   ├── model_v1.pkl
│   │   └── .gitkeep
│   ├── checkpoints/                   # Training checkpoints
│   │   └── .gitkeep
│   └── README.md
│
├── results/
│   ├── metrics/                       # Model scores
│   ├── visualizations/                # Plots & charts
│   ├── reports/                       # Analysis reports
│   └── README.md
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest config
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_integration.py
│
├── scripts/
│   ├── train.py                       # Main training script
│   ├── predict.py                     # Prediction script
│   ├── evaluate.py                    # Evaluation script
│   └── visualize.py                   # Visualization script
│
├── config/
│   ├── config.yaml                    # Main configuration
│   ├── model_config.yaml              # Model parameters
│   └── data_config.yaml               # Data config
│
├── docs/
│   ├── setup.md
│   ├── data_dictionary.md
│   ├── methodology.md
│   └── architecture.md
│
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## ⚙️ Tech Stack

### 🧑‍💻 Programming
- Python 3.x

### 📚 Libraries
- Pandas & NumPy (Data Processing)
- Matplotlib & Seaborn (Visualization)
- Scikit-learn (Machine Learning)

### 🛠 Tools
- Jupyter Notebook
- Pytest (Testing)
- Docker (Containerization)

---

## 🔄 ML Pipeline Workflow

1. **Data Loading**
   - Load raw CSV files from `/data/raw`

2. **Data Cleaning**
   - Handle missing values
   - Remove duplicates
   - Standardize formats

3. **Feature Engineering**
   - Create new features like:
     - Total medals
     - Country performance ratios
     - Year-based trends

4. **Model Training**
   - Regression Models
   - Classification Models
   - Ensemble Methods

5. **Model Evaluation**
   - Accuracy
   - Precision / Recall
   - RMSE / MAE

6. **Visualization**
   - Medal trends
   - Country comparisons
   - Performance graphs

---

## 🚀 Getting Started

### 1️⃣ Clone Repository
```bash
git clone https://github.com/XC0ID/Olympics-Summer-Winter-Analysis.git
cd Olympics-Summer-Winter-Analysis
```
### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run Training Pipeline
```bash
python scripts/train.py
```
### 5️⃣ Run Evaluation
```bash
python scripts/evaluate.py
```
### 6️⃣ Generate Visualizations
```bash
python scripts/visualize.py
```

---

## 📈 Results
- Metrics stored in: `results/metrics/`
- Visualizations stored in: `results/visualizations/`
- Reports stored in: `results/reports/`

---

## 🧪 Testing
Run all tests using:
```bash
pytest tests/
```
---
### 📚 Documentation

Detailed documentation is available in the docs/ folder:

* Setup Guide
* Data Dictionary
* Methodology
* Architecture Overview
---

### 👨‍💻 Author

**Maulik Gajera**

[![GitHub](https://img.shields.io/badge/GitHub-Connect-black?style=for-the-badge&logo=github)](https://github.com/XC0ID)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/maulik-gajera10)
[![Kaggle](https://img.shields.io/badge/Kaggle-Connect-20BEFF?style=for-the-badge&logo=kaggle)](https://kaggle.com/maulikgajera)


---



### 📜 License

This project is licensed under the **MIT License**.

---
### ⭐ Acknowledgements

* Olympic historical datasets
* Open-source ML community
* Scikit-learn contributors
---
