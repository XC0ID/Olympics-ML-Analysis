import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader      import load_all
from src.data.cleaner     import clean_medals, clean_countries
from src.features.builder import build_features
from src.evaluation.plotter import plot_medal_trends

def main():
    medals, countries = load_all()
    features = build_features(clean_medals(medals), clean_countries(countries)).dropna()
    plot_medal_trends(features, top_n=10)
    print("Saved to results/visualizations/")

if __name__ == "__main__":
    main()