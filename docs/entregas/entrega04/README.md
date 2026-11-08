# Entrega 04 · API REST en contenedor Podman (TC5061_API_Contenedor_Equipo10.pdf)

Fuente del PDF corto (unas 5 páginas) que se sube a Canvas junto con el enlace al repositorio. La
entrega es el repositorio; el PDF solo resume, en un punto por paso de la actividad, qué se hizo,
dónde vive y una captura de evidencia del contenedor respondiendo. Misma mecánica que las entregas
anteriores: `entregable.md` incluye `secciones/00-portada.md` y `secciones/01-entregable.md`, y
`make entrega04` compila a `build/` (ignorado por Git). Convenciones de escritura en
`docs/entregas/entrega01/README.md`.

```bash
make entrega04                                                    # md + html + pdf en docs/entregas/entrega04/build/
python docs/entregas/build.py docs/entregas/entrega04 --sin-pdf   # solo md + html
```

## Estado: plantilla con pendientes

El documento se escribió antes de que el contenedor llegara a `main`. Los pasos 1 a 3 (Model Registry,
API, esquemas Pydantic) describen lo que ya existe; los pasos 4 a 6 describen el contrato acordado para
el contenedor (`Containerfile` en la raíz, `.dockerignore`, imagen `mibici-api`, puerto 8000,
`make image` y `make container`, modelos cargados desde el Model Registry vía `MLFLOW_TRACKING_URI`) y
se confirman cuando se mezclen los PR de los issues #38, #39 y #40. Lo que falta está marcado con `TODO`
dentro de recuadros `<div class="pendiente">` (naranjas en el PDF); se buscan con
`grep -rn TODO docs/entregas/entrega04/secciones/` y ninguno debe quedar en el PDF final.

## Capturas

Las capturas **aún no existen**; las toma quien atiende el issue #41 desde un `main` que ya tenga el
`Containerfile`, y las guarda en `figuras/` con estos nombres exactos, porque el README del repositorio
(sección *Contenedor de la API con Podman*) enlaza estas mismas rutas:

| Archivo | Origen |
|---|---|
| `contenedor_01_build.png` | final de `podman build -t mibici-api .` (`make image`), con la imagen creada |
| `contenedor_02_health.png` | `podman ps` con el contenedor en ejecución (`make container`) y la respuesta de `curl http://localhost:8000/health` con los tres modelos |
| `contenedor_03_predict_p1.png` | `POST http://localhost:8000/predict/p1` con el ejemplo del README y su respuesta 200; debe verse que la petición va al puerto 8000 publicado por el contenedor |

En las capturas debe distinguirse que responde el contenedor: nombre de la imagen o del contenedor en
pantalla y puerto publicado. Al guardar cada captura, en `secciones/01-entregable.md` se quita su recuadro
`pendiente` y se descomenta la línea `<!-- ![Figura N. ...](../figuras/...) -->` que ya trae el pie de
figura. Las capturas de terminal pueden ser renders del texto real que imprimió cada comando, como en la
entrega 03. `logo_tec.png` es el mismo de las entregas anteriores.

## Antes de entregar

- [ ] No queda ningún `TODO` ni recuadro `pendiente` en el PDF (incluida la fecha de la portada).
- [ ] Las tres capturas existen con los nombres de la tabla y se leen con nitidez al tamaño del PDF.
- [ ] Los pasos 4 a 6 coinciden con el `Containerfile`, el `.dockerignore`, el `Makefile` y el README que están en `main`.
- [ ] La tabla de PR coincide con la pestaña *Pull requests* del repositorio.
- [ ] El PDF tiene 6 páginas o menos y se llama `TC5061_API_Contenedor_Equipo10.pdf`. La plantilla sin
      capturas ya ocupa 6 páginas (con los recuadros `pendiente`, que desaparecen); si al agregar las tres
      capturas se pasa, bajar `max-height` de las figuras en el bloque `<style>` de `entregable.md`.
