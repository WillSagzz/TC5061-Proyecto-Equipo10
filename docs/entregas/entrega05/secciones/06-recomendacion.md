# Recomendación sobre *Continuous Training*

**Se recomienda reentrenar el modelo cada mes.** MiBici publica los viajes una vez al mes, así que antes de esa fecha no hay datos nuevos con los cuales reentrenar.

El motivo es que la red crece. El `@champion` se entrenó con 371 estaciones y en septiembre operaban 468 (sección 3). El modelo no conoce una estación nueva hasta que se reentrena con sus viajes. La versión 2, entrenada con seis meses más de datos, tiene un MAE 7.5 % menor en las estaciones nuevas y 1.2 % menor en total (Tabla 4).

Otras opciones comunes son reentrenar cuando se detecta drift o cuando sube el error (Google Cloud, s.f.). Aquí ninguna de las dos señales es confiable. El reporte de Evidently no marcó drift del conjunto en P1, porque solo cambiaron 2 de sus 11 columnas, y aun así reentrenar ayudó. El MAE del `@champion` subió de 0.908 en su periodo de prueba a 1.017 en septiembre, pero el de la versión reentrenada fue de 1.004: casi toda esa subida se debe al mes, no al modelo.

Una versión reentrenada no reemplaza a la vigente de forma automática, porque puede salir peor (Evidently AI, s.f.). Primero se comparan las dos sobre el mes nuevo, como en la sección 5. La nueva queda vigente solo si su MAE es menor, si el intervalo de confianza de la diferencia no incluye el cero y si no empeora en las estaciones conocidas ni en las nuevas. Si después falla, se regresa a la anterior moviendo el alias `champion`.

Hoy estos pasos se ejecutan a mano. El siguiente paso es programarlos para que corran solos cada mes: descargar los datos, generar el reporte de drift, reentrenar, comparar las dos versiones y registrar el resultado en MLflow. Reentrenar P1 tarda un minuto. La decisión de cambiar la versión vigente se deja a una persona.

Lo mismo aplica a P2, que tiene drift en las mismas dos variables. En P3 conviene recalcular los grupos, porque se formaron sin las estaciones que abrieron después de febrero de 2026.

## Referencias

Evidently AI. (s.f.). *When to retrain machine learning models*. https://learn.evidentlyai.com/ml-observability-course/module-4-designing-effective-ml-monitoring/when-to-retrain-ml-models

Google Cloud. (s.f.). *MLOps: Continuous delivery and automation pipelines in machine learning*. https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
