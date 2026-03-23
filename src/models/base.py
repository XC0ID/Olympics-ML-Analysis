import joblib
from pathlib import Path
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def build(self): pass

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Saved: {path}")

    def load(self, path: str):
        self.model = joblib.load(path)
        return self