import pandas as pd
from sklearn.ensemble import IsolationForest

# Обрабатываем данные через IsolationForest
class ISOForestDetector(IsolationForest):
    
    def __init__(self, **params):
        super().__init__(**params)
    
    def _add_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[['value', 'year', 'month', 'time_idx']]
    
    def fit(self, X: pd.DataFrame, y=None):
        return super().fit(self._add_features(X))
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return super().predict(self._add_features(X)) == -1
    
    def fit_predict(self, X: pd.DataFrame, y=None) -> pd.Series:
        self.fit(X)
        return self.predict(X)
    