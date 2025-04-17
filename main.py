import hydra
import pandas as pd
from omegaconf import DictConfig
from isoforest_detector import ISOForestDetector
from preprocessor import DataPreprocessor
from prophet_detector import ProphetDetector


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    # Загрузка данных
    preprocessor = DataPreprocessor(cfg.data.input_path)
    df = preprocessor.process()
    
    # Детекция аномалий
    anomalies = pd.DataFrame(index=df.index)
    for metric in cfg.data.metrics:

        # Для каждой метрики прогоняем еще раз модель по-новой
        prophet_detector = ProphetDetector(**cfg.models.prophet)
        iso_detector = ISOForestDetector(**cfg.models.isoforest)
        
        metric_df = df[[metric]].rename(columns={metric: 'value'})
        metric_df = metric_df.join(df[['year', 'month', 'time_idx']])
        
        # Обучение и предсказание
        prophet_anomalies = prophet_detector.fit_predict(metric_df)
        iso_anomalies = iso_detector.fit_predict(metric_df)
        
        anomalies[metric] = prophet_anomalies & iso_anomalies
    
    # Сохранение результатов
    df.join(anomalies.add_prefix('anomaly_')).to_excel(cfg.data.output_path)

if __name__ == "__main__":
    main()
    