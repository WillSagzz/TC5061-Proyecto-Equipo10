# Repositorio y reproducción

Repositorio: <https://github.com/WillSagzz/TC5061-Proyecto-Equipo10>. Resultados obtenidos con el *commit* `TODO` de `main` y la versión de datos `TODO`.

<div class="pendiente">

TODO. Un párrafo: servidor de *tracking* usado (local con `sqlite:///mlflow.db` o el compartido en Azure); qué declara el `MLproject` (entorno `python_env.yaml`, puntos de entrada `ingest` y `train`, parámetros y sus valores por defecto desde `config.py`); y que todas las corridas de este reporte se lanzaron con `mlflow run`, verificable en las etiquetas `mlflow.source.type = PROJECT` y `mlflow.project.entryPoint` de cada corrida.

</div>

Desde un clon limpio:

```bash
# TODO: copiar del README los comandos exactos
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

mlflow run . -e ingest

mlflow run . -e train -P question=p1 -P model=TODO -P TODO=TODO   # modelo en producción de P1
mlflow run . -e train -P question=p2 -P model=TODO -P TODO=TODO   # modelo en producción de P2
mlflow run . -e train -P question=p3 -P model=TODO -P TODO=TODO   # modelo en producción de P3
```
