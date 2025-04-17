from prophet import Prophet
from sklearn.base import BaseEstimator, TransformerMixin

# Обрабатываем через Prophet для интеграции в sklearn-пайплайны
class ProphetDetector(BaseEstimator, TransformerMixin):
    
    def __init__(self, **params):
        self.model = Prophet(**params)
        
    def fit(self, X, y=None):
        train_df = X.reset_index().rename(columns={'month_dt': 'ds', 'value': 'y'})
        self.model.fit(train_df)
        return self
        
    def predict(self, X):
        forecast = self.model.predict(X)
        return (X['value'] > forecast['yhat_upper']) | (X['value'] < forecast['yhat_lower'])
    