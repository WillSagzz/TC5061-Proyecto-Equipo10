# Entrega 01 · Propuesta de proyecto (TC5061_Propuesta_Proyecto_Equipo10.pdf)

Fuente en Markdown del PDF que se sube a Canvas. El documento se escribe en **varias secciones**
(una por tema, en `secciones/`) y se compila en un solo Markdown, HTML y PDF con el script
genérico `docs/entregas/build.py`.

## Estructura

```text
entrega01/
├── README.md            <- este archivo
├── entregable.md        <- archivo MAESTRO: metadatos + orden de las secciones ({!secciones/xx.md!})
├── secciones/           <- una sección por archivo, numeradas para mantener el orden
│   ├── 00-portada.md
│   ├── 01-introduccion.md        (enlace al repositorio, preguntas del proyecto)
│   ├── 02-formulacion.md         (P1, P2, P3)
│   ├── 03-ml-canvas.md           (imagen del canvas; contenido por bloque en TODO)
│   ├── 04-datos.md               (rango, fuentes, tamaño, variables, condiciones de uso, partición)
│   ├── 05-pipeline.md
│   ├── 06-modelos-metricas.md
│   ├── 07-arquitectura.md        (TODO: diagrama + tabla de herramientas)
│   ├── 08-experimento-base.md    (TODO: diseño, resultados, capturas de MLflow)
│   ├── 09-repositorio.md         (TODO: estructura, buenas prácticas, flujo Git, integrantes)
│   └── 10-referencias.md         (TODO: APA 7)
├── figuras/             <- capturas e imágenes exclusivas de esta entrega (logo, capturas de MLflow)
├── build/               <- SALIDA generada (ignorada por Git): .md compilado, .html y .pdf
└── TC5061_Propuesta_Proyecto_Equipo10.pdf  <- copia del PDF final que sí se versiona (la deja el script)
```

Las imágenes compartidas con el repositorio (ML Canvas, diagrama de arquitectura) viven en
`references/`, como pide la actividad, y se referencian con rutas relativas desde la sección.

## Cómo compilar

Desde la raíz del repositorio:

```bash
make entrega01                                        # md + html + pdf en docs/entregas/entrega01/build/
python docs/entregas/build.py docs/entregas/entrega01 --sin-pdf   # solo md + html (revisión rápida)
python docs/entregas/build.py docs/entregas/entrega01 --abrir     # genera y abre el pdf
```

Requisitos: las dependencias de `requirements.txt` (`markdown-it-py`, `PyYAML`) y un navegador
Chromium instalado (Microsoft Edge o Google Chrome) para imprimir el PDF. Si el navegador no se
detecta solo, define la variable de entorno `NAVEGADOR_PDF` con la ruta al ejecutable. El diagrama
Mermaid se renderiza con una librería descargada de un CDN, por lo que la compilación del PDF
necesita conexión a internet.

## Convenciones de escritura

- **Una sección por archivo.** Cada archivo empieza con un encabezado de nivel 1 (`# Título`) y usa
  `##`/`###` para subsecciones. La tabla de contenido se genera a partir de `#` y `##`.
- **Orden.** Lo define `entregable.md`; para mover o agregar una sección basta con editar esa lista.
- **Saltos de página.** El comentario HTML `pagebreak` (ver `entregable.md`) fuerza un salto en el PDF.
- **Tablas y figuras numeradas.** Las tablas llevan la leyenda arriba (`**Tabla N.** ...`) y las
  figuras se escriben como imagen sola en un párrafo: `![Figura N. Leyenda.](../figuras/archivo.png)`;
  el compilador las convierte en `<figure>` con pie de figura.
- **Pendientes.** Lo que falta por escribir o decidir se marca con la palabra `TODO` (buscable con
  `grep -rn TODO secciones/`); las explicaciones largas van dentro de `<div class="pendiente"> ... </div>`
  (con líneas en blanco antes y después del contenido), que se ve como un recuadro naranja en el PDF.
  **Todo TODO debe desaparecer antes de la entrega final.** Las secciones que no existían en el Word
  original (arquitectura, experimento base, repositorio, referencias y el texto del canvas) están en TODO.
- **Diagramas.** Bloques ```` ```mermaid ```` se renderizan en el HTML/PDF y también en GitHub. Cualquier
  archivo de texto puede incluirse con `{!ruta!}`, por ejemplo un `.mmd` dentro de un bloque mermaid.
- **Citas.** Formato APA 7 en el texto (Autor, año) y lista completa en `10-referencias.md`.

## Lista de verificación antes de entregar

- [ ] No queda ningún `TODO` ni recuadro `pendiente` en el PDF.
- [ ] Las cuatro capturas de MLflow y la captura del repositorio están en `figuras/` y descomentadas.
- [ ] `references/ml_canvas.png` está actualizado y legible, y el diagrama de arquitectura está en `references/`.
- [ ] La definición de P2 (clases, granularidad y horizonte) es la misma en todas las secciones y en `config.py`.
- [ ] La sección de experimento base tiene los resultados reales.
- [ ] El PDF generado se llama `TC5061_Propuesta_Proyecto_Equipo10.pdf`.
