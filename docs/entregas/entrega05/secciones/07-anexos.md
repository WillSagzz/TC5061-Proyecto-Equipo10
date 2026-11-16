# Anexos

## Comandos para reproducir el reporte

El código está en el repositorio: `mibici/monitoring.py` (conjuntos), `mibici/drift.py` (reporte de Evidently) y `mibici/modeling/ab_test.py` (versión alternativa y A/B testing). Los comandos se ejecutan desde la raíz, con las dependencias de `requirements.txt` instaladas y el archivo `.env` del servidor de MLflow del equipo.

```bash
make data            # descarga y limpia los 24 meses de la versión de datos
make monitor-data    # descarga septiembre de 2026 y arma los conjuntos de referencia y actual
make drift           # reportes de Evidently en reports/evidently/
make retrain         # reentrena P1 y lo registra como @challenger
make ab-test         # compara @champion y @challenger; tabla en reports/ab_test/
make entrega05       # compila este PDF
```

## Tabla de drift por variable de P2

| column | method | score | threshold | drift |
| --- | --- | --- | --- | --- |
| capacity | Wasserstein distance (normed) | 0.101 | 0.1 | True |
| station_id | Jensen-Shannon distance | 0.287 | 0.1 | True |
| departures_lag_1 | Wasserstein distance (normed) | 0.030 | 0.1 | False |
| departures_lag_8 | Wasserstein distance (normed) | 0.034 | 0.1 | False |
| departures_lag_56 | Wasserstein distance (normed) | 0.053 | 0.1 | False |
| arrivals_lag_1 | Wasserstein distance (normed) | 0.029 | 0.1 | False |
| arrivals_lag_8 | Wasserstein distance (normed) | 0.030 | 0.1 | False |
| net_flow_lag_1 | Wasserstein distance (normed) | 0.076 | 0.1 | False |
| net_flow_lag_8 | Wasserstein distance (normed) | 0.075 | 0.1 | False |
| departures_mean_3 | Wasserstein distance (normed) | 0.043 | 0.1 | False |
| interval_of_day | Jensen-Shannon distance | 0.006 | 0.1 | False |
| day_of_week | Jensen-Shannon distance | 0.036 | 0.1 | False |
| is_weekend | Jensen-Shannon distance | 0.015 | 0.1 | False |
| risk | Jensen-Shannon distance | 0.026 | 0.1 | False |

## Tabla de drift por variable de P3

| column | method | score | threshold | drift |
| --- | --- | --- | --- | --- |
| midday_pct | K-S p_value | 0.000 | 0.05 | True |
| afternoon_pct | K-S p_value | 0.009 | 0.05 | True |
| night_pct | K-S p_value | 0.002 | 0.05 | True |
| weekdays_pct | K-S p_value | 0.025 | 0.05 | True |
| balance_afternoon | K-S p_value | 0.004 | 0.05 | True |
| morning_pct | K-S p_value | 0.120 | 0.05 | False |
| log_daily_use_per_dock | K-S p_value | 0.570 | 0.05 | False |
| balance_morning | K-S p_value | 0.931 | 0.05 | False |
