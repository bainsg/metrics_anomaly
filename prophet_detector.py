import pandas as pd
from prophet import Prophet
from sklearn.base import BaseEstimator, TransformerMixin

# Обрабатываем через Prophet для интеграции в sklearn-пайплайны
class ProphetDetector(BaseEstimator, TransformerMixin):
    
    def __init__(self, **params):
        self.params = params  # Сохраняем параметры конфигурации
        self._reset_model()   # Инициализируем модель при создании
        
    def _reset_model(self):
        """Сбрасывает состояние модели Prophet"""
        self.model = Prophet(**self.params)
        
    def fit(self, X: pd.DataFrame, y=None):
        """Обучение на новых данных со сбросом состояния"""
        self._reset_model()  # Важно: создаем новую модель для каждого обучения
        train_df = X.reset_index().rename(columns={'month_dt': 'ds', 'value': 'y'})
        self.model.fit(train_df)
        return self
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Предсказание аномалий"""
        future = self.model.make_future_dataframe(periods=0, freq='ME')
        forecast = self.model.predict(future)
        
        merged = X.merge(
            forecast[['ds', 'yhat_lower', 'yhat_upper']], 
            left_index=True, 
            right_on='ds'
        )
        
        return (merged['value'] > merged['yhat_upper']) | (merged['value'] < merged['yhat_lower'])
    
    def fit_predict(self, X: pd.DataFrame, y=None) -> pd.Series:
        """Комбинированный метод обучения и предсказания"""
        self.fit(X)
        return self.predict(X)
   