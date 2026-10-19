---
# Metadatos del documento compilado. `archivo_salida` define el nombre del .md/.html/.pdf en build/.
titulo: "TC5061 · Proyecto MiBici: propuesta y base reproducible · Equipo 10"
archivo_salida: TC5061_Propuesta_Proyecto_Equipo10
---

<!--
Archivo maestro de la Entrega 01. Define el ORDEN de las secciones; el contenido vive en secciones/.
Cada linea {!ruta!} se sustituye por el archivo indicado al compilar con:

    python docs/entregas/build.py docs/entregas/entrega01

Marcadores disponibles: el marcador "toc" entre dobles llaves genera la tabla de contenido y el
comentario HTML "pagebreak" (ver mas abajo) fuerza un salto de pagina en el PDF.
-->

{!secciones/00-portada.md!}

## Contenido {.sin-toc}

{{toc}}

<!-- pagebreak -->

{!secciones/01-introduccion.md!}

{!secciones/02-formulacion.md!}

<!-- pagebreak -->

{!secciones/03-ml-canvas.md!}

<!-- pagebreak -->

{!secciones/04-datos.md!}

<!-- pagebreak -->

{!secciones/05-pipeline.md!}

<!-- pagebreak -->

{!secciones/06-modelos-metricas.md!}

<!-- pagebreak -->

{!secciones/07-arquitectura.md!}

<!-- pagebreak -->

{!secciones/08-experimento-base.md!}

<!-- pagebreak -->

{!secciones/09-repositorio.md!}

<!-- pagebreak -->

{!secciones/10-referencias.md!}
