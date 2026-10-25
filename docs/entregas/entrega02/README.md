# Entrega 02 · Reporte de experimentos con MLflow (TC5061_Reporte_MLFlow_Equipo10.pdf)

Fuente del PDF que se sube a Canvas. Misma mecánica que la Entrega 01: una sección por archivo en
`secciones/`, orden en `entregable.md`, compilación con `make entrega02` (salida en `build/`, ignorada
por Git). Convenciones de escritura en `docs/entregas/entrega01/README.md`.

Cada sección corresponde a un punto del reporte que pide la actividad:

| Sección | Punto del reporte |
|---|---|
| `01-reproduccion.md` | Enlace al repositorio y comandos de `mlflow run` |
| `02-comparacion.md` | Tabla comparativa por pregunta contra el *baseline*; análisis y justificación del modelo elegido |
| `03-ciclo-de-vida.md` | Modelos registrados y esquema de ciclo de vida con su justificación |
| `04-capturas.md` | Capturas: *nested runs*, artefactos de una corrida candidata, tres modelos en producción con linaje |
| `05-referencias.md` | Referencias APA 7 |

Lo que falta está marcado con `TODO` (`grep -rn TODO secciones/`) y recuadros `pendiente`; nada de eso
debe quedar en el PDF. Las figuras y capturas van en `figuras/` con los nombres que ya aparecen
comentados en `02-comparacion.md` y `04-capturas.md`; si no se usan las figuras 1 a 3, renumerar las
capturas. Nombres de corridas, etiquetas, modelos registrados y comandos se copian del código, del README
y de la interfaz de MLflow.
