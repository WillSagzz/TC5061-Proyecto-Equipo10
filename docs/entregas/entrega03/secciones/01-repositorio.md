# Repositorio y cómo ejecutar los chequeos

Repositorio: <https://github.com/WillSagzz/TC5061-Proyecto-Equipo10>. Es el mismo repositorio de las entregas anteriores (`TC5061-Proyecto-Equipo10`); esta entrega corresponde a la Actividad 7 y deja integrados en él las pruebas automatizadas con PyTest, la validación de datos con DeepChecks y los chequeos básicos de seguridad, de forma que cualquier integrante, el revisor de un *Pull Request* o GitHub Actions los ejecute con un solo comando.

Las entregas 01 y 02 dejaron la ingesta reproducible con datos versionados (`datapackage.json`), el módulo de *feature engineering* (`mibici/features.py`), el MLflow Project con el *baseline* y los candidatos de cada pregunta, y el Model Registry con el modelo en producción de P1, P2 y P3. Lo que esta entrega agrega encima de eso se resume en la Tabla 1. El principio que guía el diseño es el que proponen Huyen (2022) y Wilson (2022): las pruebas y las validaciones de datos forman parte del código del proyecto, se ejecutan en cada cambio y no dependen de que alguien tenga los datos reales en su máquina.

**Tabla 1.** Qué agrega esta entrega y dónde vive cada pieza en el repositorio.

| Pieza | Dónde vive | Qué hace |
|---|---|---|
| Pruebas unitarias y de reproducibilidad | `tests/` (10 archivos, 123 pruebas) | Prueban las funciones clave de limpieza, *feature engineering*, catálogo de modelos, umbral de P2, *baselines* y entrenamiento, con casos normales y límite; tres pruebas comprueban que entrenar dos veces con la misma semilla da el mismo resultado. |
| Validación de datos con DeepChecks | `mibici/validation.py`, umbrales en `mibici/config.py`, pruebas en `tests/test_data_validation.py` | Suites de integridad (P1, P2 y P3) y de validación *train-test* con *drift* (P1 y P2) sobre las tablas de modelado; exporta cinco reportes HTML a `reports/deepchecks/`. |
| Prácticas de seguridad | `.gitignore`, `.env.example`, `security/vulnerabilidades-aceptadas.txt` | Mantienen fuera de Git los datos, los modelos, las salidas de MLflow, el `.env` y cualquier credencial; documentan los avisos de `pip-audit` aceptados con su justificación. |
| Análisis estático de seguridad | `pyproject.toml`, sección `[tool.bandit]` | Configuración de `bandit`: exclusiones y la única regla omitida (`B101` dentro de `tests/`). |
| Auditoría de dependencias | `requirements.txt`, `security/vulnerabilidades-aceptadas.txt` | Versiones fijas, actualizadas a las que corrigen los avisos de `pip-audit`. |
| Un solo comando | `Makefile`: `test`, `check-data`, `security`, `security-report`, `check` | `make check` encadena `lint`, `test` y `security`. |
| Integración continua | `.github/workflows/checks.yml` | Cuatro *jobs* (`lint`, `test`, `bandit`, `pip-audit`) en cada *Pull Request* y en cada *push* a `main`. |
| Documentación | `README.md`: secciones *Validación de calidad de datos y pruebas automatizadas* y *Seguridad del repositorio* | Instrucciones, tablas de lo que prueba cada archivo, hallazgos atendidos y evidencia de ejecución. |

## Ejecutar todo desde un clon limpio

Las pruebas y las validaciones trabajan con datos inventados o con el generador sintético de `validation.py`, así que no hace falta descargar los 758 MB de CSV de MiBici ni tener un archivo `.env`. Con Python 3.12 instalado:

```bash
git clone https://github.com/WillSagzz/TC5061-Proyecto-Equipo10.git
cd TC5061-Proyecto-Equipo10
py -3.12 -m venv .venv                   # macOS / Linux: python3.12 -m venv .venv
.venv\Scripts\activate                   # macOS / Linux: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

make check                               # lint + test + security con un solo comando
```

`make check` es la suma de tres objetivos del `Makefile`; cada uno se puede correr por separado y tiene un equivalente sin `make` (en Windows, `make` viene con Git Bash o se instala con `winget install GnuWin32.Make`):

```bash
make lint          # ruff format --check && ruff check
make test          # python -m pytest -v
make security      # python -m bandit -c pyproject.toml -r mibici infra/scripts docs/entregas/build.py tests
                   # python -m pip_audit -r requirements.txt --ignore-vuln <cada id de security/vulnerabilidades-aceptadas.txt>
```

Dos objetivos más completan la evidencia de esta entrega:

```bash
make check-data SYNTHETIC=1    # python -m mibici.validation --synthetic: 5 suites de DeepChecks y 5 reportes HTML en reports/deepchecks/
make check-data                # lo mismo con los datos reales de data/processed/ (requiere make data antes)
make security-report           # bandit en HTML y CSV y pip-audit en Markdown, escritos en security/ (fuera de Git)
```

Cada objetivo termina con código de salida distinto de cero si algo falla: una prueba roja, una condición de DeepChecks que no se cumple, un hallazgo de `bandit` o un aviso de `pip-audit` que no esté aceptado. Eso es lo que permite que GitHub Actions ejecute exactamente los mismos comandos y marque el *Pull Request* en rojo (sección 4).
