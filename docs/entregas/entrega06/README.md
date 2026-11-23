# Entrega 06 · Microservicios, CI/CD y FinOps (TC5061_Operacion_FinOps_Equipo10.pdf)

Fuente del reporte que se sube a Canvas: la orquestación con Docker Compose, el *pipeline* de
CI/CD, los resultados de la prueba de carga con Locust y el análisis FinOps. Misma mecánica que
las entregas anteriores: `entregable.md` incluye los archivos de `secciones/` y `make entrega06`
compila a `build/` (ignorado por Git). Convenciones de escritura en
`docs/entregas/entrega01/README.md`.

```bash
make entrega06                                                    # md + html + pdf en docs/entregas/entrega06/build/
python docs/entregas/build.py docs/entregas/entrega06 --sin-pdf   # solo md + html
```

**Formato que pide la actividad:** Arial 11, interlineado 1.15 y de 3 a 5 cuartillas sin contar
anexos. La fuente y el interlineado están en el bloque `<style>` de `entregable.md`. Las
cuartillas que cuentan son las de las secciones 01 a 04; la portada y los anexos van en páginas
aparte.

## Secciones

| Archivo | Contenido |
|---|---|
| `00-portada.md` | Portada, con el enlace al repositorio |
| `01-orquestacion.md` | Servicios y configuración de `docker-compose.yml` |
| `02-pipeline.md` | *Jobs* del *workflow* de GitHub Actions y la corrida exitosa |
| `03-pruebas-carga.md` | Escenario de Locust, gráficas, uso de CPU y memoria y áreas de mejora |
| `04-finops.md` | Costo por inferencia y optimizaciones |
| `05-anexos.md` | Comandos para reproducir, resultados por endpoint, precios de referencia y referencias |

## Datos de la prueba de carga

`make load-test` escribe sus resultados en `reports/locust/`, que está fuera de Git. Los de la
corrida que usa el reporte están copiados aquí:

- `figuras/locust_01_desempeno.png` y `figuras/locust_02_recursos.png`: las dos gráficas.
- `anexos/resumen_por_escalon.csv` y `anexos/resumen_por_endpoint.csv`: las tablas de las
  secciones 03 y 04 salen de estos archivos.
- `anexos/prueba_configuraciones.csv`: la comparación de configuraciones de la API de la sección 04.

Si se repite la prueba, hay que copiar de nuevo esos archivos y actualizar las cifras de las
secciones 03 y 04, que dependen de la máquina donde corre.

## Antes de entregar

- [ ] El PDF no tiene secciones sin escribir y la portada lleva la fecha de entrega.
- [ ] Las secciones 01 a 04 ocupan de 3 a 5 cuartillas.
- [ ] El PDF contiene las gráficas de Locust, el cálculo del costo por inferencia y el enlace al
      repositorio.
- [ ] Las cifras de las secciones 03 y 04 coinciden con los CSV de `anexos/`.
- [ ] La corrida más reciente del *workflow* en `main` está en verde.
- [ ] El PDF se llama `TC5061_Operacion_FinOps_Equipo10.pdf`.
