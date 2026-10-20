// Servidor MLflow compartido del equipo en Azure.
//
// Un contenedor con la imagen oficial de MLflow en Azure Container Apps, los metadatos en
// Azure Database for PostgreSQL (Flexible Server) y los artefactos en Azure Blob Storage. El
// acceso de las personas es con la autenticación básica de MLflow (usuario y contraseña); el
// servidor entra a Blob con una identidad administrada, sin claves de almacenamiento.
//
// Se despliega con `make infra-deploy` (ver infra/README.md). Los secretos llegan como
// parámetros seguros desde infra/.secrets.env (fuera de Git) a través de main.bicepparam.
targetScope = 'resourceGroup'

// -----------------------------------------------------------------------------------------
// Parámetros
// -----------------------------------------------------------------------------------------
@description('Región de todos los recursos.')
param location string = resourceGroup().location

@description('Nombre del equipo; prefijo de los recursos.')
@minLength(3)
@maxLength(20)
param teamName string = 'mibici-equipo10'

@description('Tag de la imagen oficial ghcr.io/mlflow/mlflow. La variante -full incluye psycopg2, azure-storage-blob, azure-identity y mlflow[auth]; existe desde v3.10.0 y trae auth desde v3.14.0.')
param mlflowImageTag string = 'v3.17.1-full'

@description('vCPU del contenedor. Combinaciones válidas del plan Consumption: 0.25/0.5Gi, 0.5/1Gi, 0.75/1.5Gi, 1/2Gi, ...')
param containerCpu string = '0.5'

@description('Memoria del contenedor (ver containerCpu).')
param containerMemory string = '1Gi'

@description('Object ID (Entra) de quien administra el despliegue. Si se indica, recibe Storage Blob Data Reader para poder listar los artefactos con `az storage blob list --auth-mode login`. Vacío para omitir.')
param adminPrincipalId string = ''

@description('Contraseña del usuario administrador de PostgreSQL. Solo [A-Za-z0-9] para no codificarla en la URI.')
@secure()
@minLength(16)
param postgresPassword string

@description('Contraseña inicial del usuario `admin` de MLflow (mínimo 12 caracteres). Solo se usa la primera vez que arranca el servidor con la base vacía.')
@secure()
@minLength(12)
param mlflowAdminPassword string

@description('Clave secreta de Flask para la protección CSRF de la autenticación básica.')
@secure()
@minLength(32)
param flaskSecretKey string

// -----------------------------------------------------------------------------------------
// Nombres
// -----------------------------------------------------------------------------------------
var suffix = uniqueString(resourceGroup().id)
var teamSlug = toLower(replace(teamName, '-', ''))
// Storage: 3-24 caracteres, solo minúsculas y dígitos. 2 + 9 + 13 = 24.
var storageAccountName = 'st${take(teamSlug, 9)}${suffix}'
var artifactsContainerName = 'mlflow-artifacts'
var postgresServerName = 'pg-${teamName}-${suffix}'
var postgresAdminUser = 'mlflow'
var postgresDatabaseName = 'mlflow'
var logAnalyticsName = 'log-${teamName}'
var environmentName = 'cae-${teamName}'
var identityName = 'id-${teamName}-mlflow'
var appName = 'mlflow'

// Roles integrados de Azure
var storageBlobDataContributorRoleId = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
var storageBlobDataReaderRoleId = '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'

// -----------------------------------------------------------------------------------------
// Observabilidad: Log Analytics (obligatorio para el entorno de Container Apps)
// -----------------------------------------------------------------------------------------
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// -----------------------------------------------------------------------------------------
// Identidad administrada del servidor (acceso a Blob sin claves)
// -----------------------------------------------------------------------------------------
resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

// -----------------------------------------------------------------------------------------
// Artefactos: Storage account + contenedor privado
// -----------------------------------------------------------------------------------------
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false // solo identidad (RBAC): no existen claves que filtrar
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    publicNetworkAccess: 'Enabled'
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource artifactsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: artifactsContainerName
  properties: {
    publicAccess: 'None'
  }
}

resource blobContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, identity.id, storageBlobDataContributorRoleId)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      storageBlobDataContributorRoleId
    )
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource adminBlobReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(adminPrincipalId)) {
  name: guid(storage.id, adminPrincipalId, storageBlobDataReaderRoleId)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      storageBlobDataReaderRoleId
    )
    principalId: adminPrincipalId
    principalType: 'User'
  }
}

// -----------------------------------------------------------------------------------------
// Metadatos: PostgreSQL Flexible Server (Burstable B1ms)
// -----------------------------------------------------------------------------------------
resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: postgresServerName
  location: location
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    version: '16'
    administratorLogin: postgresAdminUser
    administratorLoginPassword: postgresPassword
    storage: {
      storageSizeGB: 32
      autoGrow: 'Disabled'
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
    network: {
      publicNetworkAccess: 'Enabled'
    }
    authConfig: {
      passwordAuth: 'Enabled'
      activeDirectoryAuth: 'Disabled'
    }
  }
}

resource postgresDatabase 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: postgres
  name: postgresDatabaseName
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

// 0.0.0.0-0.0.0.0 es la regla especial "permitir servicios de Azure" (la Container App no
// tiene IP fija en el plan Consumption). SSL sigue siendo obligatorio.
resource postgresFirewallAzure 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2024-08-01' = {
  parent: postgres
  name: 'AllowAllAzureServicesAndResourcesWithinAzureIps'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// -----------------------------------------------------------------------------------------
// Cómputo: entorno y Container App
// -----------------------------------------------------------------------------------------
resource containerEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

var appFqdn = '${appName}.${containerEnv.properties.defaultDomain}'
var backendStoreUri = 'postgresql://${postgresAdminUser}:${postgresPassword}@${postgres.properties.fullyQualifiedDomainName}:5432/${postgresDatabaseName}?sslmode=require'
var artifactsDestination = 'wasbs://${artifactsContainerName}@${storage.name}.blob.${az.environment().suffixes.storage}'

// Arranque del servidor, en UNA sola línea a propósito: un salto de línea de Windows (CRLF)
// dentro de un script multilínea rompe bash ("set: pipefail: invalid option name"). Escribe el
// ini de basic-auth (los usuarios viven en la misma base PostgreSQL que el tracking) y lanza
// `mlflow server`. Container Apps inyecta variables de OpenTelemetry que desvían el tracing de
// MLflow; se quitan. La cola de trabajos del servidor (/dev/shm) se apaga por variable de entorno.
var startupScript = 'set -eu; unset OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT OTEL_EXPORTER_OTLP_METRICS_ENDPOINT OTEL_EXPORTER_OTLP_LOGS_ENDPOINT; printf "[mlflow]\\ndefault_permission = READ\\ndatabase_uri = %s\\nadmin_username = admin\\n" "$MLFLOW_BACKEND_STORE_URI" > /tmp/basic_auth.ini; exec mlflow server --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" --artifacts-destination "$MLFLOW_ARTIFACTS_DESTINATION" --host 0.0.0.0 --port 5000 --workers 1 --app-name basic-auth'

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 5000
        transport: 'auto'
        allowInsecure: false
      }
      secrets: [
        {
          name: 'backend-store-uri'
          value: backendStoreUri
        }
        {
          name: 'mlflow-admin-password'
          value: mlflowAdminPassword
        }
        {
          name: 'flask-secret-key'
          value: flaskSecretKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'mlflow'
          image: 'ghcr.io/mlflow/mlflow:${mlflowImageTag}'
          resources: {
            cpu: json(containerCpu)
            memory: containerMemory
          }
          command: [
            '/bin/bash'
            '-c'
          ]
          args: [
            startupScript
          ]
          env: [
            {
              name: 'MLFLOW_BACKEND_STORE_URI'
              secretRef: 'backend-store-uri'
            }
            {
              name: 'MLFLOW_ARTIFACTS_DESTINATION'
              value: artifactsDestination
            }
            {
              name: 'MLFLOW_AUTH_CONFIG_PATH'
              value: '/tmp/basic_auth.ini'
            }
            {
              name: 'MLFLOW_AUTH_ADMIN_PASSWORD'
              secretRef: 'mlflow-admin-password'
            }
            {
              name: 'MLFLOW_FLASK_SERVER_SECRET_KEY'
              secretRef: 'flask-secret-key'
            }
            {
              // Desde MLflow 3.5 el servidor rechaza peticiones cuya cabecera Host no esté en
              // esta lista (403 "Invalid Host header"). /health está exento.
              name: 'MLFLOW_SERVER_ALLOWED_HOSTS'
              value: '${appFqdn},${appFqdn}:*,localhost,localhost:*,127.0.0.1,127.0.0.1:*'
            }
            {
              // El middleware bloquea los POST del navegador cuya cabecera Origin no esté en la
              // lista (solo confía en localhost por defecto): la propia UI fallaba con 403
              // "Cross-origin request blocked" en runs/search y traces/metrics.
              name: 'MLFLOW_SERVER_CORS_ALLOWED_ORIGINS'
              value: 'https://${appFqdn}'
            }
            {
              name: 'MLFLOW_SERVER_ENABLE_JOB_EXECUTION'
              value: 'false'
            }
            {
              // DefaultAzureCredential usa esta identidad para leer y escribir en Blob
              name: 'AZURE_CLIENT_ID'
              value: identity.properties.clientId
            }
          ]
          probes: [
            {
              type: 'Startup'
              httpGet: {
                path: '/health'
                port: 5000
              }
              periodSeconds: 10
              failureThreshold: 30
            }
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 5000
              }
              periodSeconds: 30
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
  dependsOn: [
    postgresDatabase
    postgresFirewallAzure
    artifactsContainer
    blobContributor
  ]
}

// -----------------------------------------------------------------------------------------
// Salidas
// -----------------------------------------------------------------------------------------
output mlflowUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
output postgresHost string = postgres.properties.fullyQualifiedDomainName
output storageAccountName string = storage.name
output artifactsContainer string = artifactsContainerName
output containerAppName string = app.name
