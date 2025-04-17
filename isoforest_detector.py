from sklearn.ensemble import IsolationForest

# Обрабатываем данные через IsolationForest
class ISOForestDetector(IsolationForest):
    
    def __init__(self, **params):
        super().__init__(**params)
    
    def fit(self, X, y=None):
        return super().fit(self._add_features(X))
    
    def _add_features(self, X):
        return X[['value', 'year', 'month', 'time_idx']]