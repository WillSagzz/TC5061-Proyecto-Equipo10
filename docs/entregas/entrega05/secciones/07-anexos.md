# Anexos

## Anexo A. Comandos para reproducir el reporte

Se ejecutan desde la raíz del repositorio, con las dependencias de `requirements.txt` instaladas y el archivo `.env` del servidor de MLflow del equipo.

```bash
make data            # descarga y limpia los 24 meses de la versión de datos
make monitor-data    # descarga septiembre de 2026 y arma los conjuntos de referencia y actual
make drift           # reportes de Evidently en reports/evidently/
make retrain         # reentrena P1 y lo registra como @challenger
make ab-test         # compara @champion y @challenger; tabla en reports/ab_test/
make entrega05       # compila este PDF
```

<div class="pendiente">

**TODO (#55).** Anexo B, con las tablas y figuras de drift de P2 y P3 que no quepan en el cuerpo.

</div>
