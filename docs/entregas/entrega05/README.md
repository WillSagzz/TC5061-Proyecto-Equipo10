# Entrega 05 · Monitoreo, drift y A/B testing (TC5061_Drift_ABTest_Equipo10.pdf)

Fuente del reporte que se sube a Canvas: el reporte de drift con las gráficas de Evidently, su
interpretación, el A/B testing de las dos versiones del modelo y la recomendación sobre
*Continuous Training*. Misma mecánica que las entregas anteriores: `entregable.md` incluye los
archivos de `secciones/` y `make entrega05` compila a `build/` (ignorado por Git). Convenciones de
escritura en `docs/entregas/entrega01/README.md`.

```bash
make entrega05                                                    # md + html + pdf en docs/entregas/entrega05/build/
python docs/entregas/build.py docs/entregas/entrega05 --sin-pdf   # solo md + html
```

**Formato que pide la actividad:** Arial 11, interlineado 1.15 y de 3 a 5 cuartillas sin contar
anexos. La fuente y el interlineado están en el bloque `<style>` de `entregable.md` y solo aplican
a esta entrega. Las cuartillas que cuentan son las de las secciones 01 a 06; la portada y los
anexos van en páginas aparte.

## Secciones

Una sección por archivo, para que cada parte se escriba en su propio PR sin conflictos:

| Archivo | Contenido | Issue |
|---|---|---|
| `00-portada.md` | Portada, con el enlace al repositorio | #56 |
| `01-conjuntos.md` | Conjuntos de referencia y actual: periodos, por qué se eligieron y filas por pregunta | #56 |
| `02-drift.md` | Reporte de drift con las figuras de Evidently | #55 |
| `03-interpretacion.md` | Variables afectadas e implicaciones para el modelo | #55 |
| `04-version-alternativa.md` | Versión nueva en el Model Registry | #59 |
| `05-ab-testing.md` | Tabla comparativa y versión vigente | #59 |
| `06-recomendacion.md` | Recomendación sobre *Continuous Training* | #60 |
| `07-anexos.md` | Comandos para reproducir y drift por variable | #57 |

Las capturas del reporte de Evidently están en `figuras/` (`logo_tec.png` es el mismo de las
entregas anteriores).

## Antes de entregar

- [ ] El PDF no tiene secciones sin escribir y la portada lleva la fecha de entrega.
- [ ] Las secciones 01 a 06 ocupan de 3 a 5 cuartillas. Si se pasan, bajar `max-height` de las
      figuras en el bloque `<style>` de `entregable.md` o mover figuras a los anexos.
- [ ] El PDF contiene las gráficas de Evidently, la tabla comparativa del A/B, la recomendación
      final y el enlace al repositorio.
- [ ] Las cifras coinciden con los reportes de `reports/evidently/`, con las tablas de
      `reports/ab_test/` y con el Model Registry.
- [ ] El PDF se llama `TC5061_Drift_ABTest_Equipo10.pdf`.
