# Model Registry y ciclo de vida

**Tabla 5.** Modelos registrados y su linaje.

| Pregunta | Nombre registrado | Versión | Alias | Descripción | Corrida de origen | *Commit* | Versión de datos |
|----------|-------------------|---------|-------|-------------|-------------------|----------|------------------|
| P1 | `mibici-equipo10-p1-demanda` | TODO | `champion` | TODO | TODO | TODO | TODO |
| P2 | `mibici-equipo10-p2-riesgo` | TODO | `champion` | TODO | TODO | TODO | TODO |
| P3 | `mibici-equipo10-p3-grupos` | TODO | `champion` | TODO | TODO | TODO | TODO |

<div class="pendiente">

TODO. Un párrafo: esquema usado (aliases `champion`/`challenger` o etapas) y por qué; cómo se promueve y se revierte una versión.

</div>

Carga del modelo en producción por nombre y alias:

```python
# TODO: fragmento ejecutado y su salida
import mlflow

modelo = mlflow.pyfunc.load_model("models:/mibici-equipo10-p1-demanda@champion")
print(modelo.metadata.signature)
```
