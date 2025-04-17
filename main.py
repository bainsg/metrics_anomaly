import hydra
from omegaconf import DictConfig
import pandas as pd
from src.processing.preprocessor import DataPreprocessor
from src.detectors import ProphetDetector, ISOForestDetector

@hydra.main(version_base=None, config_path="../config", config_name="settings")
def main(cfg: DictConfig) -> None:

    # Загружаем и обрабатываем данные
    preprocessor = DataPreprocessor(cfg.data.input_path)
    df, dates = preprocessor.process()
    
    prophet_detector = ProphetDetector(**cfg.models.prophet)
    iso_detector = ISOForestDetector(**cfg.models.isoforest)
    
    # Производим поиск аномалий для всех метрик
    anomalies = pd.DataFrame(index=dates)
    for metric in cfg.data.metrics:
        # Подготовка данных
        metric_df = df[[metric]].rename(columns={metric: 'value'})
        
        # Поиск через Prophet
        prophet_anomalies = prophet_detector.fit_predict(metric_df)
        
        # Поиск через Isolation Forest
        iso_anomalies = iso_detector.fit_predict(metric_df)
        
        # Комбинируем модели
        anomalies[metric] = prophet_anomalies & iso_anomalies
    
    # Выгружаем в excel
    df.join(anomalies.add_prefix('anomaly_')).to_excel(cfg.data.output_path)

if __name__ == "__main__":
    main()