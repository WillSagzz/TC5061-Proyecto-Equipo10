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
