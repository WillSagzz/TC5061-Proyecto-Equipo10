# Entrega 03 · Resumen de pruebas y chequeos de seguridad (TC5061_Tests_Seguridad_Equipo10.pdf)

Fuente del PDF corto (unas 5 páginas) que se sube a Canvas junto con el enlace al repositorio. La
entrega es el repositorio; el PDF solo resume, en un punto por instrucción de la actividad, qué se hizo,
dónde vive y una captura de evidencia. Misma mecánica que las entregas anteriores: `entregable.md`
incluye `secciones/00-portada.md` y `secciones/01-entregable.md`, y `make entrega03` compila a
`build/` (ignorado por Git). Convenciones de escritura en `docs/entregas/entrega01/README.md`.

Las capturas de `figuras/` también las usa el README del repositorio (subsección *Evidencia de ejecución
de las pruebas y los chequeos de seguridad*), por eso no se mueven ni se renombran:

| Archivo | Origen |
|---|---|
| `pytest_01_resumen.png` | `python -m pytest -v` (resumen final) |
| `deepchecks_01_consola.png` | `python -m mibici.validation --synthetic` (consola) |
| `bandit_01_resultado.png` | `python -m bandit -c pyproject.toml -r mibici infra/scripts docs/entregas/build.py tests` |
| `pip_audit_02_despues.png` | `pip-audit -r requirements.txt` con los avisos aceptados omitidos |
| `github_02_checks_pr30.png` | pestaña *Checks* del PR #30, recortada al panel de jobs |

Las capturas de terminal son renders del texto real que imprimió cada comando. Para actualizarlas basta
con reemplazar el PNG con el mismo nombre y volver a compilar:

```bash
make entrega03                                                    # md + html + pdf en docs/entregas/entrega03/build/
python docs/entregas/build.py docs/entregas/entrega03 --sin-pdf   # solo md + html
```

Antes de entregar: el PDF tiene 6 páginas o menos, los números coinciden con las capturas y con la
última corrida de `make check`, y la tabla de PR coincide con la pestaña *Pull requests*.
