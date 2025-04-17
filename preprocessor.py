import pandas as pd
from pydantic import validate_arguments
from typing import Tuple

# Создаем класс для предобработки временных рядов
class DataPreprocessor:

    @validate_arguments
    def __init__(self, data_path: str):
        self.df = pd.read_excel(data_path, parse_dates=['month_dt'])
        self._validate_data()
        
    def _validate_data(self) -> None:
        required_columns = {'month_dt', 'revenue', 'MAU'}
        if not required_columns.issubset(self.df.columns):
            raise ValueError("Missing required columns")
    
    # Задаем основной пайплайн обработки данных
    def process(self) -> pd.DataFrame:
        self.df = (
            self.df
            .set_index('month_dt')
            .sort_index()
            .pipe(self._add_features)
            .pipe(self._handle_missing)
        )
        return self.df
    
    # Докидываем фичи
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(
            year=df.index.year,
            month=df.index.month,
            time_idx=(df.index.year - df.index.year.min()) * 12 + df.index.month
        )
    
    # Если есть пропуски, обрабатываем их
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.interpolate(method='linear').ffill().bfill()
    