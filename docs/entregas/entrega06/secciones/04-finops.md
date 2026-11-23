# Análisis FinOps

## Costo por inferencia

El costo se estima con una instancia de referencia que cubre lo que midió la prueba: 2 CPU para la API y menos de 1 GB de memoria entre los dos contenedores. Es una AWS EC2 `t3.medium` (2 vCPU y 4 GB) bajo demanda en `us-east-1`: USD 0.0416 por hora, que son USD 30.37 por un mes de 730 horas (Amazon Web Services, s.f.). El anexo la compara con instancias equivalentes de otros proveedores.

El costo por predicción es el precio por hora dividido entre las predicciones por hora. De cada 11 peticiones de la prueba, 10 son predicciones.

**Tabla 4.** Costo por millón de predicciones según cuánto se usa la instancia.

| Uso de la instancia | Predicciones/s | Predicciones al mes | USD por millón |
|---|---:|---:|---:|
| Saturada (50 usuarios o más) | 34.4 | 90.4 millones | 0.34 |
| Con p95 de 190 ms (25 usuarios) | 19.6 | 51.5 millones | 0.59 |
| Con p95 de 17 ms (10 usuarios) | 8.9 | 23.5 millones | 1.29 |
| Demanda de MiBici | 0.15 | 0.4 millones | 76 |

Calcular una predicción cuesta muy poco: USD 0.34 por millón con la instancia saturada. Lo que encarece cada predicción es el tiempo que la instancia pasa encendida sin trabajo. MiBici opera 468 estaciones. P1 pronostica cada estación en cada una de las 20 horas de servicio del día siguiente, que son 9,360 predicciones al día. P2 revisa cada estación cada 3 horas, que son 3,744 al día. P3 se consulta una vez al mes. En total son unas 400 mil predicciones al mes, el 0.4 % de lo que la instancia puede atender. Con ese uso cada predicción cuesta USD 0.000076, más de 200 veces lo que cuesta con la instancia saturada.

Si la demanda crece, el costo por predicción baja hasta que la instancia se llena. A partir de ahí se agregan réplicas iguales y el costo por predicción ya no cambia: cada réplica atiende 51 millones de predicciones al mes con un p95 de 190 ms.

## Optimizaciones

**1. Un hilo por predicción y dos *workers*.** Para medir la primera área de mejora de la sección 3 se repitió la prueba durante 90 segundos, con 100 usuarios, en cuatro configuraciones de la API.

**Tabla 5.** Desempeño de la API con 100 usuarios según su configuración.

| Configuración de la API | Peticiones/s | p50 (ms) | p95 (ms) | Memoria API (MiB) |
|---|---:|---:|---:|---:|
| Actual: 1 *worker* | 39.0 | 1,200 | 3,900 | 231 |
| 2 *workers* | 30.5 | 1,100 | 6,800 | 415 |
| 1 *worker*, 1 hilo por predicción | 71.2 | 310 | 1,000 | 195 |
| 2 *workers*, 1 hilo por predicción | 92.8 | 18 | 330 | 403 |

Con un hilo por predicción y dos *workers*, la misma instancia atiende 2.4 veces más peticiones y el p95 baja de 3.9 a 0.33 segundos. Agregar *workers* sin limitar los hilos empeora el resultado, porque más hilos compiten por las mismas 2 CPU. Con 250 usuarios esta configuración se satura en 93 peticiones por segundo, así que el costo con la instancia saturada baja de USD 0.34 a 0.14 por millón de predicciones. El cambio son dos variables de entorno del servicio `api`: `OMP_NUM_THREADS=1` y `API_WORKERS=2`.

**2. Una instancia más chica.** La API usó 235 MiB de memoria (unos 410 con dos *workers*) y MLflow 390 MiB: menos de 1 GB de los 4 GB de la instancia. Una `t3.small` tiene las mismas 2 vCPU y 2 GB por USD 0.0208 la hora, la mitad.

**3. Encender el servicio solo cuando se usa.** Las predicciones llegan en rondas: la de P2 cada 3 horas y la de P1 una vez al día. A 20 predicciones por segundo, una ronda de P2 tarda medio minuto y la de P1 ocho minutos. Si la instancia se enciende 10 minutos para cada una de las 8 rondas del día, trabaja unas 40 horas al mes en lugar de 730, y AWS cobra por segundo. El sistema levanta en menos de 40 segundos con `docker compose up`.

**Tabla 6.** Costo al mes con la demanda de MiBici (400 mil predicciones).

| Configuración | Horas encendida | USD al mes | USD por millón |
|---|---:|---:|---:|
| Actual: `t3.medium` todo el mes | 730 | 30.37 | 76 |
| `t3.small` todo el mes | 730 | 15.18 | 38 |
| `t3.small` solo durante las rondas | 40 | 0.84 | 2.1 |

Las optimizaciones 2 y 3 bajan el costo de hoy, porque reducen el tiempo sin trabajo. La 1 importa cuando la demanda crezca: cada réplica atenderá 2.4 veces más peticiones antes de necesitar otra.
