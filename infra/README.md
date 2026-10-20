# Servidor MLflow compartido del equipo en Azure

Infraestructura como código del **MLflow Tracking Server** que usa el equipo para que los
*runs* de todos los integrantes queden en una sola instancia, con la misma interfaz que el
modo local. Es la materialización del componente 7 de la arquitectura de la Entrega 01.

Este documento tiene dos partes: **cómo lo usa un integrante** (no necesita Azure) y **cómo lo
despliega y mantiene quien administra la suscripción**.

| Pieza | Recurso de Azure | Notas |
|---|---|---|
| Servidor MLflow | **Azure Container Apps** (plan Consumption, 0.5 vCPU / 1 GiB, una réplica siempre encendida) | Imagen oficial `ghcr.io/mlflow/mlflow:<versión>-full`; HTTPS con certificado incluido; sin Dockerfile ni registro propio. |
| Metadatos (experimentos, *runs*, usuarios) | **Azure Database for PostgreSQL Flexible Server** (Burstable B1ms, PostgreSQL 16, 32 GB) | SSL obligatorio; firewall solo para servicios de Azure. |
| Artefactos (modelos, figuras, tablas) | **Azure Blob Storage** (Standard LRS, contenedor privado `mlflow-artifacts`) | El servidor entra con una **identidad administrada** (rol *Storage Blob Data Contributor*); la cuenta tiene las claves compartidas deshabilitadas. |
| Logs | Log Analytics | Obligatorio para el entorno de Container Apps; `make infra-logs`. |

Todo vive en un grupo de recursos propio del proyecto, `rg-mibici-equipo10`, en
`mexicocentral` (Querétaro): es la región más cercana al equipo y sus precios son equivalentes
a los de Estados Unidos (PostgreSQL B1ms 13.65 USD/mes frente a 14.02 en `centralus`; Container
Apps y Blob entre 8 % y 24 % más caros, pero son centavos). La suscripción tiene restringido
PostgreSQL Flexible Server en `eastus`, `eastus2`, `southcentralus` y `westus2`;
`az postgres flexible-server list-skus -l <región>` lo muestra. Costo total aproximado: 20 a
30 USD al mes, cubierto por el crédito mensual de la suscripción.

**Qué no está en el repositorio, a propósito:** la URL del servidor, la lista de usuarios,
las contraseñas y los secretos del despliegue. El repositorio es público; todo eso se entrega
en privado y vive en archivos locales excluidos de Git (`.env`, `infra/users.txt`,
`infra/.users.env`, `infra/.secrets.env`).

---

## Parte 1 · Para integrantes: conectarse y usar el servidor

No necesitas cuenta de Azure ni ninguna herramienta extra: solo el entorno de Python del
proyecto (README principal, sección *Instalación*) y tres datos que te entrega en privado quien
administra el servidor: la **URL**, tu **usuario** y tu **contraseña inicial**.

### 1. Crear tu `.env`

En la raíz del repositorio, copia la plantilla y llena las tres variables:

```bash
cp .env.example .env
```

```dotenv
MLFLOW_TRACKING_URI=https://mlflow.<lo-que-te-dieron>.mexicocentral.azurecontainerapps.io
MLFLOW_TRACKING_USERNAME=tu-usuario
MLFLOW_TRACKING_PASSWORD=tu-contraseña
```

`.env` está en `.gitignore`: nunca entra a Git. Ninguna credencial de Azure hace falta, porque
los artefactos (modelos, figuras, CSV) viajan a través del servidor hacia Blob Storage.

### 2. Probar la conexión

```bash
make infra-check        # o: python infra/scripts/check_server.py
```

Crea un *run* de prueba en el experimento `smoke-test` con un parámetro, una métrica y un
artefacto, lo vuelve a leer y descarga el artefacto. Si termina con `Todo OK`, funcionan la
autenticación, la base de datos y el almacenamiento de artefactos.

### 3. Registrar experimentos

Nada cambia respecto al modo local: con el `.env` presente, `make train` (o
`python -m mibici.modeling.train p1`) registra en el servidor, en el experimento
`mibici-equipo10`. Cada *run* lleva la etiqueta `usuario` con tu nombre de usuario, además de
`git_commit`, `pregunta`, `equipo` y `tipo_modelo`, así que en la interfaz se ve quién corrió
qué.

### 4. Ver los resultados

Abre la URL del servidor en el navegador: pide usuario y contraseña (los mismos del `.env`) y
muestra la interfaz de MLflow de siempre, con los *runs* de todo el equipo. Es la misma
interfaz para quien revisa la entrega, con un usuario de solo lectura.

### 5. Cambiar tu contraseña

```bash
make mlflow-password    # o: python infra/scripts/change_password.py
```

Lee tu usuario y contraseña actual del `.env`, pide la nueva en la terminal (mínimo 12
caracteres, sin mostrarla) y la actualiza en el servidor. Después cambia
`MLFLOW_TRACKING_PASSWORD` en tu `.env`. No hay pantalla para esto en la interfaz web. Si la
olvidas, pide a quien administra que la restablezca (`make infra-reset-password`).

### 6. Alternar entre local y servidor

El modo lo decide solo `MLFLOW_TRACKING_URI`:

| Quieres | Haz |
|---|---|
| Trabajar en local un rato (iterar, depurar) | `MLFLOW_TRACKING_URI=sqlite:///mlflow.db make train` (una variable en la línea de comandos gana sobre el `.env`) |
| Trabajar en local hasta nuevo aviso | Renombra `.env` a `.env.nube` y vuelve a renombrarlo cuando quieras el servidor |
| Ver la base local | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |

Local y servidor son almacenes distintos: un *run* local nunca aparece en el servidor. El flujo
previsto es iterar en local y correr el experimento definitivo contra el servidor.

### 7. Problemas comunes

| Síntoma | Causa y solución |
|---|---|
| `401 Unauthorized` | Usuario o contraseña incorrectos en `.env`, o `.env` no está en la raíz del repositorio. |
| `403 Permission denied` al crear un *run* | Tu usuario es de solo lectura o el experimento es nuevo y aún no te dieron `EDIT`: pide a quien administra que corra `make infra-users`. |
| `UnicodeEncodeError` en Windows | La consola abre en cp1252 y MLflow imprime emojis. Los scripts de `infra/` ya lo corrigen; para `make train` usa `set PYTHONIOENCODING=utf-8` (cmd) o `$env:PYTHONIOENCODING="utf-8"` (PowerShell). |
| La URL no responde | El servidor arranca en 1 o 2 minutos tras un redeploy; si persiste, avisa a quien administra (`make infra-logs`). |

---

## Parte 2 · Para quien administra: desplegar y mantener

Requisitos: [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) con una cuenta
Owner o Contributor en la suscripción del equipo, `make`, `bash` y `openssl` (vienen con Git
Bash en Windows) y el entorno de Python del proyecto. Solo una persona necesita esto.

### Primer despliegue, paso a paso

```bash
az login                      # 1. sesión en la suscripción del equipo
make infra-prereqs            # 2. instala Bicep y la extensión containerapp; registra los proveedores
make infra-secrets            # 3. genera infra/.secrets.env (contraseña de PostgreSQL, de admin y clave CSRF)
make infra-plan               # 4. (opcional) what-if: lista lo que se va a crear, sin crear nada
make infra-deploy             # 5. crea rg-mibici-equipo10 y los 12 recursos; imprime la URL (10-15 min)
cp infra/users.example.txt infra/users.txt   # 6. escribe los usuarios reales (archivo local, no se versiona)
make infra-users              # 7. crea los usuarios, el experimento mibici-equipo10 y los permisos
```

Al terminar el paso 7, las contraseñas iniciales están en `infra/.users.env` (fuera de Git).
Guarda `infra/.secrets.env` y `infra/.users.env` en un gestor de contraseñas: son el único
lugar donde están.

### Entregar las credenciales

A cada integrante, por un canal privado (nunca por el repositorio ni por un *issue*): la URL
(`make infra-url`), su usuario y su contraseña inicial. Pídele que siga la Parte 1 y que la
cambie con `make mlflow-password`. La contraseña del usuario de solo lectura para quien
califica va en el documento de la entrega o por el canal del curso; después de la calificación
se restablece o se borra.

### Usuarios y permisos

`infra/users.txt` es la fuente: `usuario,rol` por línea, rol `editor` (ve todo y registra
*runs*) o `lector` (solo lectura). `make infra-users` es idempotente: crea los que falten,
crea el experimento `mibici-equipo10` si no existe y otorga `EDIT` a los editores sobre todos
los experimentos existentes.

| Tarea | Cómo |
|---|---|
| Agregar un integrante | Añadir la línea a `users.txt` y `make infra-users`; su contraseña aparece en `.users.env` |
| Quitar un integrante | Borrar la línea y `make infra-users PRUNE=1` |
| Alguien olvidó su contraseña | `make infra-reset-password USUARIO=nombre`; la nueva se agrega a `.users.env` |
| Alguien creó un experimento nuevo | `make infra-users` (quien lo crea es `MANAGE`; el resto queda en `READ` hasta correr esto) |
| Rotar la contraseña de `admin` | Con las credenciales actuales: `MLFLOW_TRACKING_URI=... MLFLOW_TRACKING_USERNAME=admin MLFLOW_TRACKING_PASSWORD=... python infra/scripts/change_password.py` y actualizar `infra/.secrets.env` (la variable del contenedor solo se usa la primera vez) |

Crear usuarios exige las credenciales de `admin`: nadie puede registrarse solo.

### Mantener la infraestructura

| Tarea | Cómo |
|---|---|
| Ver la URL | `make infra-url` |
| Ver logs del servidor | `make infra-logs` |
| Cambiar de versión de MLflow | `mlflowImageTag` en [`main.bicepparam`](main.bicepparam) (variante `-full`, ≥ `v3.14.0-full`) y `make infra-deploy` |
| Más CPU o memoria | `containerCpu` y `containerMemory` en `main.bicepparam` (pares válidos: 0.5/1Gi, 0.75/1.5Gi, 1/2Gi, ...) y `make infra-deploy` |
| Cualquier otro cambio | Editar [`main.bicep`](main.bicep), `make infra-plan` y luego `make infra-deploy`; Bicep es idempotente y los *runs* y usuarios persisten en PostgreSQL |
| Apagar todo | `make infra-destroy` borra el grupo de recursos completo (pide confirmación). Los *runs* se pierden: exporta antes lo que haga falta |

`make infra-plan` muestra siempre una "modificación" en la Container App aunque no haya
cambios: el *what-if* no puede resolver el dominio del entorno ni el *client id* de la
identidad antes de desplegar y los imprime como expresiones. Es ruido conocido.

### Decisiones de configuración que conviene conocer

- **Imagen oficial sin personalizar.** La variante `-full` incluye el *driver* de PostgreSQL,
  el cliente de Blob, `azure-identity` y la autenticación básica desde la versión 3.14. El
  cliente 3.4.0 del repositorio habla con un servidor 3.x más nuevo, que es la dirección de
  compatibilidad que MLflow soporta.
- **`MLFLOW_SERVER_ALLOWED_HOSTS`.** Desde MLflow 3.5 el servidor rechaza con 403 las
  peticiones cuya cabecera `Host` no esté en la lista; el template la llena con el FQDN de la
  Container App y lo resuelve solo.
- **`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`.** La cola de trabajos del servidor (MLflow
  ≥ 3.15) usa `/dev/shm`; no se necesita para *tracking* y así se evita un modo de falla
  conocido en contenedores con poca memoria compartida.
- **Variables `OTEL_*`.** Container Apps las inyecta y desvían el *tracing* de MLflow; el
  script de arranque las elimina.
- **`default_permission = READ`** en la autenticación básica; los permisos `EDIT` son
  explícitos por usuario y experimento (`create_users.py`). La base de usuarios es la misma
  PostgreSQL del *tracking* (tablas propias), así que sobrevive a redeploys.
- **Firewall de PostgreSQL** con la regla `0.0.0.0` ("servicios de Azure") porque la Container
  App del plan Consumption no tiene IP fija. Para conectarte con `psql` desde tu máquina agrega
  una regla temporal con tu IP desde el portal.
- **Sin VNet.** Endpoints públicos con TLS, autenticación y RBAC bastan para un proyecto de
  curso; la guía oficial de MLflow usa VNet y queda como mejora posible.

### Por qué esta arquitectura y no otra

- **MLflow propio y no Azure Machine Learning.** Microsoft no ofrece un servidor MLflow
  administrado; su alternativa es usar un *workspace* de Azure ML como *tracker*, que muestra
  los *runs* en Azure ML Studio (no en la interfaz de MLflow), da soporte oficial hasta MLflow
  2.16 y exige una identidad de Entra por integrante. El curso pide capturas de la interfaz de
  MLflow y el repositorio fija `mlflow==3.4.0`.
- **Container Apps + PostgreSQL + Blob** es la combinación que documenta la propia guía de
  MLflow para Azure y la que usan los despliegues publicados en 2025 y 2026. App Service solo
  aparece en guías de 2020 a 2023 y tiene un límite de memoria compartida que rompe MLflow
  3.15 en adelante; Container Instances no tiene HTTPS; AKS es excesivo para un contenedor.
- **Sin claves de Azure en ningún `.env`.** Los artefactos pasan por el servidor (*proxied
  artifacts*), así que las laptops solo necesitan la URL, un usuario y una contraseña.

## Referencias

- MLflow, *Deploying MLflow Tracking Server to Azure*: <https://mlflow.org/docs/latest/self-hosting/deploy-to-cloud/azure/>
- MLflow, *Basic HTTP authentication*: <https://mlflow.org/docs/latest/self-hosting/security/basic-http-auth/>
- MLflow, *Network security (allowed hosts)*: <https://mlflow.org/docs/latest/self-hosting/security/network/>
- MLflow, *SDK and server compatibility*: <https://mlflow.org/docs/latest/self-hosting/migration/>
- Microsoft Learn, *MLflow and Azure Machine Learning*: <https://learn.microsoft.com/azure/machine-learning/concept-mlflow>
- Microsoft Learn, *Comparing container options in Azure*: <https://learn.microsoft.com/azure/container-apps/compare-options>
- Microsoft Learn, *Billing in Azure Container Apps*: <https://learn.microsoft.com/azure/container-apps/billing>
