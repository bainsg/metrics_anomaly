import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from prophet import Prophet
from sklearn.ensemble import IsolationForest

# Считываем исходные данные
df = pd.read_excel(r'metrics.xlsx', parse_dates=['month_dt'])
df = df.set_index('month_dt').sort_index()
print("Данные за период:", df.index.min(), "-", df.index.max())

# Заполняем пропуски линейной интерполяцией
df = df.interpolate(method='linear')

# Создаем фичи для временных рядов
df['year'] = df.index.year
df['month'] = df.index.month
df['time_idx'] = (df.index.year - df.index.year.min()) * 12 + df.index.month

# Создаем функцию поиска аномалий через Prophet
def detect_anomalies_prophet(metric_series, threshold=0.15):

    # Подготавливаем данные
    prophet_df = metric_series.reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # Обучаем модель
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    model.fit(prophet_df)
    
    # Делаем прогноз
    future = model.make_future_dataframe(periods=0, freq='ME')
    forecast = model.predict(future)
    
    # Производим слияние с исходными данными
    merged = prophet_df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    merged['anomaly_prophet'] = (merged['y'] > merged['yhat_upper']) | (merged['y'] < merged['yhat_lower'])
    
    return merged.set_index('ds')['anomaly_prophet']

# Создаем функцию поиска аномалий через Isolation Forest
def detect_anomalies_isoforest(dataframe, contamination=0.05):

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    features = dataframe[['value', 'year', 'month', 'time_idx']]
    dataframe['anomaly_iso'] = model.fit_predict(features) == -1
    return dataframe

# Обрабатываем имеющиеся метрики
metrics = ['revenue', 'MAU', 'PU', 'ARPPU', 'avg_retention_1', 'avg_retention_7']
anomaly_results = []

for metric in metrics:
    # Поиск через Prophet
    prophet_anomalies = detect_anomalies_prophet(df[metric])
    
    # Поиск через Isolation Forest
    metric_df = df[[metric]].copy()
    metric_df.columns = ['value']
    metric_df = metric_df.assign(
        year=df['year'],
        month=df['month'],
        time_idx=df['time_idx']
    )
    iso_anomalies = detect_anomalies_isoforest(metric_df)
    
    # Теперь комбинируем модели (аномалия в обеих моделях)
    combined = pd.concat([
        prophet_anomalies.rename('prophet'),
        iso_anomalies['anomaly_iso'].rename('iso')
    ], axis=1)
    combined['final_anomaly'] = combined['prophet'] & combined['iso']
    
    anomaly_results.append(combined['final_anomaly'].rename(metric))

# Создаем общую таблицу с аномалиями
anomalies_df = pd.concat(anomaly_results, axis=1)

# Строим графики для визуализации
def plot_metric_with_anomalies(metric_name):
    fig = go.Figure()
    
    # Основной график метрик
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[metric_name],
        mode='lines',
        name=metric_name
    ))
    
    # Аномалий
    anomalies = df[metric_name][anomalies_df[metric_name]]
    fig.add_trace(go.Scatter(
        x=anomalies.index,
        y=anomalies,
        mode='markers',
        marker=dict(color='red', size=10),
        name='Anomaly'
    ))
    
    fig.update_layout(
        title=f'Аномалии в метрике {metric_name}',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white'
    )
    fig.show()

for metric in metrics:
    plot_metric_with_anomalies(metric)

# Добавляем флаги аномалий в исходную таблицу
df = pd.concat([df, anomalies_df.add_prefix('anomaly_')], axis=1)

df.to_excel('metrics_with_anomalies.xlsx')

report = df.filter(like='anomaly_').sum().reset_index()
report.columns = ['metric', 'anomaly_count']
report['last_month'] = df.iloc[-1][metrics].values
print("\nОтчет по аномалиям:")
print(report)