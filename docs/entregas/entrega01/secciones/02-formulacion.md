# Formulación del problema

Para cada pregunta de negocio se establece qué se predice o agrupa, su granularidad, el horizonte temporal y, en el caso de P2, la aproximación del riesgo ante la falta de datos de inventario en tiempo real. Con ello, cada formulación se vincula directamente con las decisiones del operador de flota, del equipo de rebalanceo y del área de planeación urbana.

## P1. ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? (Regresión)

**¿Qué se predice, con qué granularidad y horizonte?** Se predice la cantidad total de viajes iniciados (salidas de bicicletas) como un valor numérico. La granularidad espacial es por estación individual y la granularidad temporal es de una hora. El horizonte de predicción es de 24 horas hacia adelante, pronosticando al cierre de cada día la demanda hora por hora del día siguiente.

**¿Por qué es relevante para la operación de MiBici?** Permite anticipar el volumen de actividad que tendrá cada punto de la red en el Área Metropolitana de Guadalajara, identificando con antelación las horas pico de saturación y las horas valle de baja actividad para dimensionar los recursos diarios del sistema sin interrumpir el servicio al ciudadano.

**¿Quién usaría el resultado y qué decisión tomaría con él?** El operador del sistema utilizaría estas predicciones para planear el tamaño de la flota activa de bicicletas que debe estar disponible en calle, programar las ventanas de mantenimiento preventivo de bicicletas y anclajes durante los horarios de menor demanda esperada, y organizar la asignación de turnos del personal operativo.

## P2. ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? (Clasificación)

**¿Qué se predice, con qué granularidad y horizonte?** Se predice una de tres categorías: *riesgo de vaciarse*, *riesgo de llenarse* o *sin riesgo*. La granularidad espacial es por estación individual y la granularidad temporal es de intervalos de 3 horas (8 intervalos por día; cada predicción habla de un bloque de 3 horas, por ejemplo: estación 25, intervalo de 9:00 a 12:00, clasificación: riesgo de vaciarse). El horizonte de predicción es de 3 horas hacia adelante: al cierre de cada intervalo se pronostica el riesgo de cada estación en el intervalo siguiente, usando solo los viajes ocurridos hasta ese momento. Así el equipo de rebalanceo puede intervenir la estación antes de que se vacíe o se llene.

<div class="pendiente">

**Pendiente de decisión del equipo (texto en revisión en el Word original).** Existe una segunda versión de la formulación de P2 que debe conciliarse con la anterior, ya que el *pipeline*, el ML Canvas y la sección de modelos usan esta variante: *"Se predice una categoría binaria ("En riesgo" o "Sin riesgo") junto con su probabilidad de ocurrencia, indicando si una estación sufrirá desabasto de bicicletas o saturación total de sus espacios de anclaje. La granularidad espacial es por estación individual, la granularidad temporal es de una hora y el horizonte de predicción es de 1 a 3 horas hacia adelante, brindando el tiempo de anticipación necesario para que el personal operativo pueda trasladarse e intervenir la estación antes de que el problema ocurra."* Hay que elegir una sola definición (tres clases en bloques de 3 h, o binaria por hora) y usarla de forma consistente en todo el documento y en `config.py`.

</div>

**Definición y justificación de la aproximación del riesgo.** Los datos de MiBici contienen el registro histórico de cada viaje y la capacidad total de anclajes de cada estación, pero no incluyen el inventario en tiempo real de cuántas bicicletas hay disponibles en cada minuto. Para resolver esta limitación de forma objetiva, el riesgo se aproxima midiendo la presión del flujo neto por intervalo en relación con el tamaño físico de cada estación:

- Para cada estación y en cada intervalo, se contabiliza el total de bicicletas que llegan (viajes que terminan ahí) y el total de bicicletas que salen (viajes que inician ahí).
- Se calcula la diferencia entre llegadas y salidas para obtener el desbalance neto de bicicletas en ese intervalo (cuántas bicicletas perdió o acumuló la estación sin compensarse por el flujo natural de los usuarios).
- Ese desbalance neto se compara contra la capacidad total de anclajes que tiene registrada la estación en el catálogo oficial.
- Cada intervalo se etiqueta en una de tres clases:
  - **Riesgo de vaciarse:** las salidas superan a las llegadas en al menos 40 % de la capacidad.
  - **Riesgo de llenarse:** las llegadas superan a las salidas en al menos 40 % de la capacidad.
  - **Sin riesgo:** la diferencia entre llegadas y salidas, sin importar si es positiva o negativa, es menor al 40 % de la capacidad.
- *Justificación metodológica:* este criterio es robusto porque evalúa la gravedad del flujo según el tamaño real de cada estación (perder 8 bicicletas en un intervalo es crítico para una estación pequeña de 12 anclajes, pero manejable para una estación grande de 40 anclajes) y evita adivinar cuántas bicicletas había al iniciar la mañana o arrastrar errores acumulados por movimientos de camionetas de rebalanceo que no están registrados en los datos abiertos.

**¿Por qué es relevante para la operación de MiBici?** Tanto una estación vacía (que impide a los usuarios iniciar un viaje) como una estación llena (que impide a los ciclistas devolver la bicicleta y cerrar su viaje en su destino) representan fallas directas en la disponibilidad del servicio público. Anticipar estos eventos permite pasar de una operación reactiva a una preventiva.

**¿Quién usaría el resultado y qué decisión tomaría con él?** El equipo de rebalanceo de bicicletas utilizaría las clasificaciones ordenadas por probabilidad de riesgo para decidir qué estaciones atender con prioridad y determinar cuántas bicicletas deben abastecer o retirar en cada punto.

## P3. ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? (Clustering)

**¿Qué se agrupa, con qué granularidad y horizonte?** Se agrupan las estaciones activas de la red en perfiles o *clusters* funcionales que comparten hábitos de uso similares. La agrupación se construye caracterizando a cada estación según qué porcentaje de sus viajes ocurre en cada intervalo del día (mañana, mediodía, tarde y noche), qué proporción de su demanda se concentra en días laborales frente a fines de semana, qué tanta rotación diaria tienen sus anclajes disponibles y si su balance habitual tiende a expulsar o a recibir viajes. La granularidad es por estación individual y el horizonte es estructural e histórico (consolidando el comportamiento típico de la estación durante todo el periodo de análisis). Una vez formados los grupos por su patrón de uso, se utilizan las coordenadas geográficas (latitud y longitud) del catálogo de estaciones para representar los *clusters* en el mapa de la ciudad y analizar su distribución territorial.

**¿Por qué es relevante para la operación de MiBici?** Permite descubrir la vocación real que tiene cada estación dentro de la movilidad del Área Metropolitana de Guadalajara (por ejemplo: distinguir estaciones residenciales de donde salen usuarios por la mañana, estaciones receptoras en zonas de oficinas o centros educativos, estaciones recreativas y estaciones periféricas de baja rotación).

**¿Quién usaría el resultado y qué decisión tomaría con él?** El área de planeación urbana y expansión del sistema utilizaría estos perfiles y su ubicación en el mapa para decidir en qué zonas de la ciudad instalar nuevas estaciones, identificar qué estaciones existentes requieren una ampliación física de anclajes por alta rotación estructural, y diseñar políticas de interconexión con otras redes de transporte público o ciclovías.
