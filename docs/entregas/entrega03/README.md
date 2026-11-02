# Entrega 03 · Reporte de pruebas y chequeos de seguridad (TC5061_Tests_Seguridad_Equipo10.pdf)

Fuente del PDF que se sube a Canvas junto con el enlace al repositorio. Misma mecánica que las
entregas anteriores: una sección por archivo en `secciones/`, orden en `entregable.md`, compilación
con `make entrega03` (salida en `build/`, ignorada por Git). Convenciones de escritura en
`docs/entregas/entrega01/README.md`.

Cada sección corresponde a un paso de la actividad y a un criterio de la rúbrica:

| Sección | Paso de la actividad / criterio |
|---|---|
| `01-repositorio.md` | Enlace al repositorio, qué agrega esta entrega y cómo correr todo con un solo comando desde un clon limpio |
| `02-pruebas-unitarias.md` | Paso 1 · Pruebas unitarias con PyTest (funciones clave, casos normales y límite, reproducibilidad) |
| `03-validacion-deepchecks.md` | Paso 2 · Validación de datos con DeepChecks (integridad, nulos, duplicados, *drift*) |
| `04-seguridad.md` | Paso 3 · Chequeos de seguridad (`.gitignore`, variables de entorno, bandit, pip-audit, CI) |
| `05-organizacion-colaboracion.md` | Pasos 4, 5 y 6 · Carpeta `tests/`, documentación en el README y colaboración en Git |
| `06-rubrica.md` | Mapa de cada criterio de la rúbrica a su evidencia |
| `07-referencias.md` | Referencias APA 7 y declaración de uso de IA |

## Figuras

Las capturas viven en `figuras/` y también las usa el README del repositorio (sección *Evidencia de
ejecución de las pruebas y los chequeos de seguridad*), por eso no se mueven ni se renombran:

| Archivo | Origen |
|---|---|
| `pytest_01_resumen.png` | `python -m pytest -v` (resumen final) |
| `deepchecks_01_consola.png` | `python -m mibici.validation --synthetic` (consola) |
| `deepchecks_02_reporte_p1_drift.png`, `deepchecks_03_reporte_p1_integridad.png` | reportes HTML exportados por el comando anterior |
| `bandit_01_resultado.png` | `python -m bandit -c pyproject.toml -r mibici infra/scripts docs/entregas/build.py tests` |
| `pip_audit_01_antes.png`, `pip_audit_02_despues.png` | `pip-audit -r requirements.txt` sobre el `requirements.txt` anterior y el actual |
| `git_01_log.png`, `git_02_shortlog.png` | `git log --oneline --graph --decorate` y `git shortlog -sn --no-merges` |
| `github_01_pull_requests.png` … `github_04_commits.png` | páginas públicas del repositorio en GitHub |

Las capturas de terminal son renders del texto real que imprimió cada comando (fondo oscuro, fuente
monoespaciada); los registros completos se conservan fuera del repositorio. Si se vuelven a correr
los comandos, basta con reemplazar el PNG con el mismo nombre.

## Cómo compilar

Desde la raíz del repositorio, con el entorno activado:

```bash
make entrega03                                                    # md + html + pdf en docs/entregas/entrega03/build/
python docs/entregas/build.py docs/entregas/entrega03 --sin-pdf   # solo md + html (revisión rápida)
python docs/entregas/build.py docs/entregas/entrega03 --abrir     # genera y abre el pdf
```

Requisitos: las dependencias de `requirements.txt` (`markdown-it-py`, `PyYAML`) y Microsoft Edge o
Google Chrome para imprimir el PDF.

## Lista de verificación antes de entregar

- [ ] No queda ningún `TODO` ni recuadro `pendiente` en el PDF (`grep -rn TODO secciones/`).
- [ ] Los números del reporte (pruebas recolectadas, suites de DeepChecks, hallazgos de bandit, avisos de pip-audit) coinciden con las capturas y con la última corrida de `make check`.
- [ ] Los comandos citados existen en el `Makefile` y en el README con el mismo nombre.
- [ ] La tabla de Pull Requests de la sección 5 coincide con la pestaña *Pull requests* del repositorio (número, título, autor, revisor).
- [ ] El PDF generado se llama `TC5061_Tests_Seguridad_Equipo10.pdf`.
