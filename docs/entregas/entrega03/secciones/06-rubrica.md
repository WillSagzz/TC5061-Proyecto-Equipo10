# Mapa de la rúbrica a la evidencia

**Tabla 15.** Criterios de la rúbrica de la Actividad 7, dónde se explica cada uno en este reporte y dónde está la evidencia en el repositorio.

| Criterio (peso) | Nivel buscado | Sección de este reporte | Evidencia en el repositorio |
|---|---|---|---|
| Pruebas unitarias con PyTest (30 %) | Cubre todas las funciones clave con casos normales y límite, incluye reproducibilidad y se ejecuta correctamente | Sección 2, Tabla 2 y Figura 1 | `tests/` (10 archivos, 123 pruebas); `make test`; *job* `test` de GitHub Actions; tabla *Pruebas unitarias* del README |
| Validación de datos con DeepChecks (25 %) | Chequeos exhaustivos que detectan y documentan valores faltantes, duplicados y *drift* | Sección 3, Tablas 3 a 5 y Figuras 2 a 4 | `mibici/validation.py`, umbrales en `config.py`, `tests/test_data_validation.py`; `make check-data`; reportes HTML en `reports/deepchecks/` |
| Chequeos de seguridad (25 %) | Herramientas y prácticas completas, sin datos sensibles y con evidencia clara de los resultados | Sección 4, Tablas 6 a 11 y Figuras 5 a 8 | `.gitignore`, `.env.example`, `pyproject.toml [tool.bandit]`, `security/vulnerabilidades-aceptadas.txt`, `requirements.txt`; `make security`, `make security-report`; *jobs* `bandit` y `pip-audit` |
| Organización y documentación (10 %) | Estructura clara, README que explica cómo ejecutar y evidencia de éxito | Sección 5, Tablas 12 y 13 y Figura 9 | Carpeta `tests/`, `Makefile` (`make check`), `.github/workflows/checks.yml`; README: *Comandos*, *Validación de calidad de datos y pruebas automatizadas*, *Seguridad del repositorio* y la subsección de evidencia con capturas |
| Colaboración (10 %) | Ramas y *commits* descriptivos, balanceados entre integrantes | Sección 5, Tabla 14 y Figuras 10 a 13 | Pestaña *Pull requests* (PR #27 a #33), historial de `main`, *ruleset* de la rama, convenciones en el README |
