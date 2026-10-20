// Parámetros del despliegue del servidor MLflow (ver infra/README.md).
// Los valores no secretos van aquí; los secretos se leen de variables de entorno que
// `make infra-deploy` exporta desde infra/.secrets.env (archivo fuera de Git).
using 'main.bicep'

// mexicocentral: la región más cercana al equipo (Querétaro) y con precio equivalente a las de
// EE. UU. La suscripción tiene restringido PostgreSQL Flexible Server en eastus, eastus2,
// southcentralus y westus2 ("Provisioning is restricted in this region").
param location = 'mexicocentral'
param teamName = 'mibici-equipo10'
param mlflowImageTag = 'v3.17.1-full'
param containerCpu = '0.5'
param containerMemory = '1Gi'

// Object ID de quien despliega; `make infra-deploy` lo obtiene con `az ad signed-in-user show`
param adminPrincipalId = readEnvironmentVariable('AZ_ADMIN_OBJECT_ID', '')

param postgresPassword = readEnvironmentVariable('PG_PASSWORD')
param mlflowAdminPassword = readEnvironmentVariable('MLFLOW_ADMIN_PASSWORD')
param flaskSecretKey = readEnvironmentVariable('MLFLOW_FLASK_SECRET_KEY')
