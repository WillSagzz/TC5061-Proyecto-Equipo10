# Capturas de la interfaz de MLflow

## Comparación de corridas con *nested runs*

![Figura 4. Experimento mibici-p1-demanda: la búsqueda P1-busqueda-hgb con sus seis corridas hijas, junto al baseline y los candidatos.](../figuras/mlflow_01_nested_runs.png)

![Figura 5. Comparación del baseline y los candidatos de P1 en la interfaz de MLflow: duración, parámetros y métricas de validación y prueba.](../figuras/mlflow_02_comparacion_p1.png)

## Artefactos de una corrida candidata

![Figura 6. Artefactos de la corrida P2-candidato-hgb_threshold: gráficas, configuración (MLproject, python_env.yaml, config.py y corrida.json), datapackage.json y el modelo registrado con su firma y un ejemplo de entrada.](../figuras/mlflow_03_artefactos.png)

## Modelos registrados en producción con su linaje

![Figura 7. Model Registry: los tres modelos registrados del equipo, con el alias champion.](../figuras/mlflow_04_registry.png)

![Figura 8. Versión champion de mibici-equipo10-p1-demanda: corrida de origen y etiquetas git_commit y version_datos.](../figuras/mlflow_05_linaje_p1.png)

![Figura 9. Versión champion de mibici-equipo10-p2-riesgo: corrida de origen y etiquetas git_commit y version_datos.](../figuras/mlflow_06_linaje_p2.png)

![Figura 10. Versión champion de mibici-equipo10-p3-grupos: corrida de origen y etiquetas git_commit y version_datos; la versión 2 tiene el alias challenger.](../figuras/mlflow_07_linaje_p3.png)

![Figura 11. Carga de los tres modelos en producción por su nombre y el alias champion.](../figuras/mlflow_08_verificacion_champion.png)
