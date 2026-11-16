# Recomendación sobre *Continuous Training*

**Sí conviene reentrenar el modelo: una vez al mes, cuando MiBici publica datos, y promoviendo la versión nueva solo si supera a la vigente.**

Los datos recientes tienen un valor que se puede medir (Huyen, 2022). La versión 2 solo se distingue de la 1 por seis meses más de entrenamiento, y con eso el MAE de septiembre baja 1.2 % en todas las estaciones y 7.5 % en las nuevas (Tabla 4). La causa es la que mostró el drift: la red pasó de 371 a 468 estaciones (sección 3), y un modelo que no se reentrena no conoce las que abren.

Los disparadores automáticos de un reentrenamiento son el calendario, la llegada de datos nuevos, una caída del desempeño y un cambio en la distribución de los datos (Google Cloud, s.f.). Aquí corresponde la llegada de datos nuevos, que equivale a un calendario mensual: MiBici publica una vez al mes y el mismo archivo trae las variables y la demanda real. Los otros dos no sirven como disparador en este proyecto. El reporte de Evidently no declara drift del conjunto en P1, porque cambian 2 de sus 11 columnas y el umbral es la mitad, y aun así reentrenar mejoró. El MAE del `@champion` subió de 0.908 en su periodo de prueba a 1.017 en septiembre, pero el de la versión reentrenada también sube, a 1.004: un umbral fijo sobre el MAE mediría el mes y no la antigüedad del modelo. El drift y el desempeño se revisan cada mes como diagnóstico.

Reentrenar no implica promover. El riesgo de reentrenar por calendario es poner en producción una versión peor que la vigente (Evidently AI, s.f.), así que cada versión nueva pasa por la validación de esta entrega: se registra como `@challenger` y se compara con el `@champion` sobre el mes recién publicado, que ninguna de las dos vio. Se promueve si tiene menor MAE, si el intervalo de confianza de la diferencia no incluye el cero y si no empeora en las estaciones conocidas ni en las nuevas. Regresar a la versión anterior es mover el alias.

Hoy ese ciclo se corre a mano. Se recomienda automatizarlo como un trabajo mensual que encadene los pasos que ya existen: ingesta del mes, reporte de drift, reentrenamiento desde cero con todos los meses anteriores (la corrida de P1 duró un minuto), comparación y registro en MLflow. Con eso el proyecto pasa del nivel 0 al nivel 1 de MLOps de Google Cloud (s.f.). La promoción se mantiene manual: una persona mueve el alias con la tabla de la comparación a la vista.

P2 tiene drift en las mismas dos variables, así que entra al mismo ciclo con su métrica principal, el F1 macro. P3 no tiene un valor real con el cual comparar dos versiones. Tiene drift en 5 de sus 8 variables y sus grupos se calcularon sin las estaciones que abrieron después de febrero de 2026, así que conviene recalcularlos y comparar la silueta con la del modelo vigente (0.173).

## Referencias

Evidently AI. (s.f.). *When to retrain machine learning models*. https://learn.evidentlyai.com/ml-observability-course/module-4-designing-effective-ml-monitoring/when-to-retrain-ml-models

Google Cloud. (s.f.). *MLOps: Continuous delivery and automation pipelines in machine learning*. https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

Huyen, C. (2022). *Designing machine learning systems*. O'Reilly Media.
