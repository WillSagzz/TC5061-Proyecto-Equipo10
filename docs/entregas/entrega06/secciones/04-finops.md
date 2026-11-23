# Análisis FinOps

## Resumen ejecutivo

La prueba de Locust alcanzó **38.67 solicitudes por segundo (RPS)** agregadas, con **0 fallos** y una latencia promedio de **1,492.66 ms**.
Sin embargo, la latencia p95 fue de **4,700 ms** y la p99 de **5,600 ms**. Lo que indica que el sistema sostuvo la carga sin errores, pero con alta variabilidad y posible saturación de recursos.

Para una plataforma de inferencia, el objetivo no es solo pagar la instancia más barata. El objetivo es mantener el **costo por predicción** bajo, manteniendo un nivel de servicio aceptable.

**Opción inicial:**
Desplegar Mibici en **DigitalOcean**, utilizando un Droplet básico de 2 vCPU / 4 GB como punto de partida para un solo entorno de bajo tráfico.
Esta es la opción más económica de las comparadas: USD 24.00/mes para 2 vCPU y 4 GB.

Esto prioriza costo y simplicidad operativa para una etapa temprana. Si el servicio predictivo de Mibici debe operar 24/7 con SLA, monitoreo, respaldo, seguridad corporativa y crecimiento sostenido, **AWS** o **Azure** son mejores plataformas estratégicas aunque su costo base de cómputo sea mayor.

## Análisis técnico y FinOps

La ausencia de fallos es algo positivo. Durante la ventana medida, el servicio respondió a todas las solicitudes. No obstante, una p95 de 4.7 segundos y una p99 de 5.6 segundos son altas para una API de predicción.
Antes de escalar horizontalmente, conviene confirmar si el cuello de botella es alguno de los siguientes factores:
- CPU
- Memoria
- Carga del modelo
- Serialización
- Red
- Contención de recursos del contenedor

Desde FinOps, esto implica que escalar réplicas sin optimizar puede aumentar el gasto sin mejorar proporcionalmente la experiencia. La primera acción debe ser identificar la utilización real de CPU, RAM, I/O y red durante la prueba. Después, aplicar right-sizing y optimización de inferencia.

### Riesgos identificados

- **Latencia de cola:** los percentiles altos sugieren que las solicitudes esperan por recursos o por el modelo. Esto puede requerir más réplicas, workers de Uvicorn/Gunicorn, batching o un modelo más eficiente.
- **Modelo `P2` más costoso:** `predict/p2` tuvo una latencia promedio de 2,357.30 ms, muy superior a `P1` (1,221.45 ms) y `P3` (999.34 ms), por lo que su huella de cómputo probablemente es mayor.
- **Falta de telemetría de recursos:** el reporte muestra rendimiento HTTP, pero no CPU, RAM, disco ni red. Se debe buscar hacer estas mediciones en el futuro, para dimensionar con mayor precisión y validar el posible ahorro de un cambio de instancia.
- **Riesgo de sobredimensionamiento:** una instancia con 2 vCPU y 4 GB puede ser suficiente para el RPS actual, pero la latencia p95/p99 indica que no hay margen cómodo para picos o crecimiento. Aqui cobra gran importancia la cantidad de usuarios concurrentes que se espera utilicen el servicio.

## Proyección de volumen

La prueba midió 38.67 RPS durante 364 segundos. Si ese nivel se sostuviera 24/7 durante un mes de 730 horas, el sistema procesaría aproximadamente **100.23 millones de solicitudes** y **91.31 millones de predicciones**.
Este no es el caso, puesto que los endpoints de `P1` y `P3` no se consultarian constantemente, pues el primero solo se utiliza para consultar datos del dia siguiente y el segundo no tiene cambios frecuentes. Solo `P2` podría requerir de consultas más frecuentes, pero solo hay 482 estaciones y el caso en el que más de 100 hagan un request al mismo tiempo es poco probable.
Aún asi se hace el análisis con este caso extremo con el objetivo de conseguir una estimación conservadora.

| Escenario | RPS de predicciones | Predicciones mensuales | Lectura FinOps |
|---|---:|---:|---|
| Prueba actual sostenida | 35.23 | 91.31 millones | Carga alta. Requiere capacidad validada y monitoreo continuo. |
| Producción moderada | 5 | 12.96 millones | Una instancia pequeña puede ser suficiente, sujeto a latencia objetivo. |
| Producción baja | 1 | 2.59 millones | Conviene una instancia más pequeña, autoscaling o apagado fuera de horario. |
| Pico estimado | 70 | 181.62 millones | Requiere al menos dos réplicas o una instancia mayor; se debe validar con otra prueba de carga. |

## Comparación de proveedores

La comparación utiliza una referencia de **2 vCPU y 4 GB de RAM**, con sistema operativo Linux y facturación bajo demanda. Los precios son de lista, no incluyen impuestos ni servicios adicionales.

| Proveedor | Instancia de referencia | vCPU / RAM | Precio mensual estimado (USD) | Costo por 1,000 solicitudes (USD) | Costo por 1,000 predicciones (USD) |
|---|---|---:|---:|---:|---:|
| **DigitalOcean** | Basic Droplet 4 GB | 2 / 4 GB | 24.00 | 0.2395 | 0.2629 |
| **Google Cloud** | e2-medium | 1 / 4 GB | 24.46 | 0.2441 | 0.2679 |
| **AWS** | EC2 t3.medium, us-east-1 | 2 / 4 GB | 30.37 | 0.3030 | 0.3326 |
| **Azure** | Standard_B2s, East US | 2 / 4 GB | 30.37 | 0.3030 | 0.3326 |

DigitalOcean es el mejor proveedor para la etapa actual de Mibici si el objetivo es minimizar costo, mantener operación simple y validar el producto. 
También factura por segundo, con un mínimo de 60 segundos o USD 0.01, lo que favorece entornos de desarrollo, staging y pruebas que no necesitan estar encendidos todo el mes.

AWS o Azure son preferibles si Mibici requiere un ecosistema empresarial más completo. AWS ofrece opciones de Reserved Instances y Spot. Para `t3.medium` en `us-east-1`, el precio efectivo de una Reserved Instance de un año con pago parcial es USD 0.025/hora, y de USD 0.0418/hora bajo demanda. Azure ofrece una instancia equivalente, `Standard_B2s`, con el mismo precio de lista de USD 0.0416/hora en East US.

Google Cloud queda en una posición intermedia. Su `e2-medium` tiene 1 vCPU y 4 GB, por lo que no es estrictamente equivalente en CPU a las opciones de 2 vCPU; aun así, su costo mensual estimado es cercano al de DigitalOcean.

| Criterio | DigitalOcean | AWS | Azure | Google Cloud |
|---|---|---|---|---|
| Costo base mensual, 2 vCPU/4 GB (USD) | 24.00 | 30.37 | 30.37 | 24.46 (1 vCPU) |
| Simplicidad operativa | Alta | Media-alta | Media-alta | Media-alta |
| Ecosistema ML/AI empresarial | Básico | Excelente | Excelente | Excelente |
| MLflow autogestionado | Sí | Sí | Sí | Sí |
| Kubernetes administrado | Sí | Sí, EKS | Sí, AKS | Sí, GKE |
| Ideal para Mibici hoy | **Sí** | Sí, si hay requisitos corporativos | Sí, si ya se usa Azure | Sí, si se usa GCP |

## Modelo de costos propuesto

El costo total no debe limitarse al servidor de inferencia. Para Mibici se recomienda presupuestar los siguientes componentes:

| Componente | Recomendación inicial | Costo estimado mensual (USD) | Notas |
|---|---|---:|---|
| Cómputo del API y modelos | 1 Droplet/VM de 2 vCPU y 4 GB | 24.00–30.37 | Según proveedor elegido. |
| MLflow backend | SQLite en disco para desarrollo; PostgreSQL administrado en producción | 0–15 | SQLite evita costo inicial, pero no es adecuado para producción multiusuario. |
| Artefactos de modelos | Object storage con ciclo de vida | 1–5 | Conservar versiones productivas; archivar experimentos antiguos. |
| Almacenamiento en bloque | 50–80 GB SSD | 5–10 | Depende del proveedor y del tamaño de imágenes, logs y modelos. |
| Observabilidad | Logs, métricas y alertas | 0–10 | Prometheus/Grafana autogestionados pueden reducir costo, a cambio de operación. |
| CI/CD y registro de imágenes | Registro y builds | 0–5 | Usar builds programados y capas de Docker optimizadas. |
| **Total inicial estimado** |  | **30–65** | Escenario de un solo entorno, sin alta disponibilidad. |

## Unit economics

Con la carga observada y una instancia de USD 24.00/mes, el costo teórico de cómputo es:

- USD 0.0002395 por solicitud.
- USD 0.2395 por cada 1,000 solicitudes.
- USD 0.2629 por cada 1,000 predicciones.
- USD 262.90 por cada millón de predicciones.

Estos valores corresponden al escenario de máxima utilización derivado de la prueba. En producción real, el costo unitario será mayor si la instancia permanece encendida con tráfico menor.
Aunque se espera un tráfico menor, esta sigue siendo una de las instancias de menor costo.

Por ejemplo, si Mibici procesa solo 1 millón de predicciones al mes pero mantiene una instancia de USD 24.00, el costo por 1,000 predicciones sube a **USD 24.00**, porque el costo fijo se distribuye entre menos volumen. Esto demuestra por qué el FinOps debe medir tanto **utilización** como **costo unitario**.

## Plan de optimización

### Acciones inmediatas

1. **Instrumentar recursos:** exponer CPU, memoria, I/O, red, latencia por endpoint y cola de solicitudes en Prometheus o un servicio equivalente.
2. **Medir por modelo:** registrar tiempo de inferencia, tamaño de entrada/salida y consumo de CPU/RAM para `p1`, `p2` y `p3`.
3. **Optimizar `predict/p2`:** al ser el endpoint con mayor latencia promedio, revisar tamaño del modelo, tipo de datos, preprocesamiento, concurrencia y necesidad de GPU.
4. **Ajustar workers:** validar el número de workers Uvicorn/Gunicorn frente a vCPU disponibles; un exceso de workers puede aumentar memoria y contención sin mejorar RPS.
5. **Reducir tamaño de imagen:** usar imágenes base ligeras, multi-stage builds y evitar dependencias innecesarias para disminuir arranque, almacenamiento y costos de red.
6. **Cachear artefactos:** cargar los modelos una sola vez al iniciar el servicio, no en cada request, y validar que MLflow no descargue artefactos repetidamente.
7. **Separar entornos:** usar dev/staging con apagado automático y producción con capacidad mínima validada.

### Optimización de capacidad

| Acción | Impacto esperado | Condición para aplicarla |
|---|---|---|
| Right-sizing a 1 vCPU / 2–4 GB | Ahorro potencial de 20–50% en cómputo | CPU promedio menor a 50–60% y latencia p95 dentro del objetivo. |
| Segunda réplica | Mejora de disponibilidad y latencia bajo picos | CPU sostenida mayor a 70% o p95 fuera de objetivo. |
| Autoscaling | Paga capacidad adicional solo durante picos | Tráfico variable y tiempo de arranque de contenedor aceptable. |
| Reserved Instances / Savings Plans | Ahorro de aproximadamente 40% en AWS para `t3.medium` de un año | Carga estable y compromiso de 1 año aceptable. |
| Spot / preemptible | Ahorro alto, pero riesgo de interrupciones | Solo para pruebas, batch o cargas tolerantes a fallos. |
| Almacenamiento por niveles | Menor costo de artefactos antiguos | Políticas de retención definidas para modelos y experimentos. |

## KPIs FinOps

| KPI | Fórmula | Meta inicial |
|---|---|---|
| Costo por predicción | Costo mensual / predicciones mensuales | Reducirlo conforme crece el volumen |
| Costo por 1,000 predicciones | (Costo mensual / predicciones) x 1,000 | Menor a USD 1.00 en producción moderada |
| Utilización promedio de CPU | CPU usada / CPU asignada | 50–70% |
| Utilización de memoria | RAM usada / RAM asignada | Menor a 80% |
| Latencia p95 | Percentil 95 de tiempo de respuesta | Definir objetivo de negocio; actualmente 4,700 ms. |
| Tasa de errores | Solicitudes fallidas / solicitudes totales | Menor a 0.1% |
| Costo de infraestructura / predicciones | Costo total / predicciones | Monitorear tendencia mensual |
| Desviación presupuestaria | Gasto real / presupuesto | Menor a 10% |

## Gobernanza FinOps

FinOps debe operar como un ciclo continuo de informar, optimizar y operar. En la etapa inicial de Mibici, el foco debe estar en visibilidad, control de gastos y decisiones de dimensionamiento basadas en evidencia.

- **Informar:** generar un reporte mensual con costo por entorno, modelo, endpoint y predicción.
- **Optimizar:** revisar tamaño de instancias, autoscaling, imágenes, almacenamiento y uso de MLflow.
- **Operar:** establecer presupuesto, alertas al 70% y 90% del presupuesto, y un proceso de aprobación para recursos nuevos.
- **Asignar costos:** etiquetar recursos por `project=mibici`, `environment=dev|staging|prod`, `component=api|mlflow|storage` y `owner`.
- **Decidir con datos:** antes de escalar, comparar el costo incremental contra la mejora medida en latencia, throughput y satisfacción del usuario.

## Conclusión

El servicio predictivo de Mibici demostró estabilidad funcional durante la prueba: procesó 14,089 solicitudes sin fallos y alcanzó 38.67 RPS.
Sin embargo, las latencias p95 y p99 indican que la infraestructura actual está cerca de su límite operativo o que existe un cuello de botella que debe diagnosticarse antes de invertir en más réplicas.

Para una primera etapa, **DigitalOcean con un Droplet de 2 vCPU / 4 GB** es la opción más rentable y sencilla, con un costo base de USD 24.00/mes.
Para un despliegue productivo con requisitos corporativos, AWS o Azure ofrecen un ecosistema más robusto, aunque con un costo base de cómputo aproximadamente 27% mayor a precio de lista.

La prioridad FinOps inmediata no es cambiar de proveedor, sino medir recursos, optimizar el modelo `P2`, validar la concurrencia del contenedor y establecer un presupuesto con alertas.
Eso permitirá escalar el servicio predictivo de Mibici con evidencia, evitando pagar por capacidad ociosa o por réplicas que no resuelven el cuello de botella real.

## Referencias

- Azure Hub. (n.d.). *Standard_B2s pricing & specs*. Recuperado el 22 de noviembre de 2026, de [https://azurehub.org/tools/vm-pricing/Standard_B2s/](https://azurehub.org/tools/vm-pricing/Standard_B2s/)
- Azure Speed. (n.d.). *Azure VM pricing: Standard_B2ls_v2*. Recuperado el 22 de noviembre de 2026, de [https://www.azurespeed.com/AzureVmPricing/Standard_B2ls_v2](https://www.azurespeed.com/AzureVmPricing/Standard_B2ls_v2)
- Azure Speed. (n.d.). *Azure VM pricing: Standard_B2pls_v2*. Recuperado el 22 de noviembre de 2026, de [https://www.azurespeed.com/AzureVmPricing/Standard_B2pls_v2](https://www.azurespeed.com/AzureVmPricing/Standard_B2pls_v2)
- Azure Speed. (n.d.). *Azure VM pricing: Standard_B2s, Southeast Asia*. Recuperado el 22 de noviembre de 2026, de [https://www.azurespeed.com/AzureVmPricing/Standard_B2s?region=southeastasia](https://www.azurespeed.com/AzureVmPricing/Standard_B2s?region=southeastasia)
- Azure Speed. (n.d.). *Azure VM pricing: Standard_B2ts_v2*. Recuperado el 22 de noviembre de 2026, de [https://www.azurespeed.com/AzureVmPricing/Standard_B2ts_v2](https://www.azurespeed.com/AzureVmPricing/Standard_B2ts_v2)
- Azure Speed. (n.d.). *Azure VM pricing: East US*. Recuperado el 22 de noviembre de 2026, de [https://www.azurespeed.com/AzureVmPricing/Regions/eastus](https://www.azurespeed.com/AzureVmPricing/Regions/eastus)
- Azure Speed. (n.d.). *Azure VM pricing: Bsv2 series*. Recuperado el 22 de noviembre de 2026, de [https://www.azurespeed.com/AzureVmPricing/Series/Bsv2](https://www.azurespeed.com/AzureVmPricing/Series/Bsv2)
- Azure Speed. (n.d.). *Azure VM pricing*. Recuperado el 22 de noviembre de 2026, de [https://www.azurespeed.com/AzureVmPricing?vcpus=128&memory=512](https://www.azurespeed.com/AzureVmPricing?vcpus=128&memory=512)
- Azure Speed. (n.d.). *Azure VM pricing*. Recuperado el 22 de noviembre de 2026, de [https://www.azurespeed.com/AzureVmPricing?vcpus=4&memory=8](https://www.azurespeed.com/AzureVmPricing?vcpus=4&memory=8)
- CloudPrice. (n.d.). *Standard_B2s specs and pricing | Azure | CloudPrice*. Recuperado el 22 de noviembre de 2026, de [https://cloudprice.net/vm/Standard_B2s](https://cloudprice.net/vm/Standard_B2s)
- Cloud2Day. (n.d.). *Standard_B2s pricing & specs: 2 vCPU, 4 GiB*. Recuperado el 22 de noviembre de 2026, de [https://cld2day.com/tools/azure-vm/standard_b2s](https://cld2day.com/tools/azure-vm/standard_b2s)
- CompareEdge. (n.d.). *Azure pricing 2026: VM rates from $0.04/hr, storage & free tier*. Recuperado el 22 de noviembre de 2026, de [https://comparedge.com/tools/azure/pricing](https://comparedge.com/tools/azure/pricing)
- DigitalOcean. (n.d.). *Droplet pricing*. Recuperado el 22 de noviembre de 2026, de [https://www.digitalocean.com/pricing/droplets](https://www.digitalocean.com/pricing/droplets)
- DigitalOcean. (n.d.). *Droplets: Scalable cloud compute starting at $4/mo*. Recuperado el 22 de noviembre de 2026, de [https://www.digitalocean.com/products/droplets](https://www.digitalocean.com/products/droplets)
- DigitalOcean. (n.d.). *Pricing*. Recuperado el 22 de noviembre de 2026, de [https://www.digitalocean.com/pricing](https://www.digitalocean.com/pricing)
- DigitalOcean. (n.d.). *VPS hosting plans: Starting at $4/mo*. Recuperado el 22 de noviembre de 2026, de [https://www.digitalocean.com/solutions/vps-hosting](https://www.digitalocean.com/solutions/vps-hosting)
- DigitalOcean. (2023, agosto 10). *Announcing new Basic Premium Droplet plans*. [https://www.digitalocean.com/blog/announcing-new-basic-premium-plans](https://www.digitalocean.com/blog/announcing-new-basic-premium-plans)
- DigitalOcean. (2023, agosto 10). *New Basic Droplet Premium CPU plans*. [https://docs.digitalocean.com/notes/2023/new-basic-droplet-premium-cpu-plans/](https://docs.digitalocean.com/notes/2023/new-basic-droplet-premium-cpu-plans/)
- DigitalOcean. (n.d.). *Droplets pricing*. Recuperado el 22 de noviembre de 2026, de [https://docs.digitalocean.com/products/droplets/details/pricing/](https://docs.digitalocean.com/products/droplets/details/pricing/)
- DoiT. (n.d.). *AWS EC2 t3.medium specs & pricing in us-east-1*. Recuperado el 22 de noviembre de 2026, de [https://www.doit.com/compute/compute/aws/us-east-1/t3.medium](https://www.doit.com/compute/compute/aws/us-east-1/t3.medium)
- Economize. (n.d.). *t3.medium pricing: $30.3680 monthly — AWS EC2*. Recuperado el 22 de noviembre de 2026, de [https://www.economize.cloud/resources/aws/pricing/ec2/t3.medium/](https://www.economize.cloud/resources/aws/pricing/ec2/t3.medium/)
- Economize. (n.d.). *Azure B2s pricing & specs*. Recuperado el 22 de noviembre de 2026, de [https://www.economize.cloud/resources/azure/pricing/virtual-machine/b2s/](https://www.economize.cloud/resources/azure/pricing/virtual-machine/b2s/)
- FinOps Foundation. (n.d.). *FinOps for AI overview*. Recuperado el 22 de noviembre de 2026, de [https://www.finops.org/wg/finops-for-ai-overview/](https://www.finops.org/wg/finops-for-ai-overview/)
- FinOps Foundation. (n.d.). *What is FinOps?* Recuperado el 22 de noviembre de 2026, de [https://www.finops.org/introduction/what-is-finops/](https://www.finops.org/introduction/what-is-finops/)
- Google Cloud. (n.d.). *Cloud Build pricing*. Recuperado el 22 de noviembre de 2026, de [https://cloud.google.com/build/pricing](https://cloud.google.com/build/pricing)
- Google Cloud. (n.d.). *Cloud Workstations pricing*. Recuperado el 22 de noviembre de 2026, de [https://cloud.google.com/workstations/pricing](https://cloud.google.com/workstations/pricing)
- Microsoft. (2026, marzo 10). *Bsv2-series sizes — Azure Virtual Machines*. [https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/general-purpose/bsv2-series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/general-purpose/bsv2-series)
- Rackspace Technology. (n.d.). *t3.medium pricing, Rackspace Spot*. Recuperado el 22 de noviembre de 2026, de [https://spot.rackspace.com/comparison/aws/ec2/t3.medium/](https://spot.rackspace.com/comparison/aws/ec2/t3.medium/)
- Rackspace Technology. (n.d.). *AWS EC2 instances*. Recuperado el 22 de noviembre de 2026, de [https://spot.rackspace.com/comparison//page/14/](https://spot.rackspace.com/comparison//page/14/)
- StackRates. (n.d.). *DigitalOcean pricing (October 2026): 31 plans from $4/mo*. Recuperado el 22 de noviembre de 2026, de [https://stackrates.ai/hosting/digitalocean/](https://stackrates.ai/hosting/digitalocean/)
- Amazon Web Services. (n.d.). *Amazon EC2 on-demand instance pricing*. Recuperado el 22 de noviembre de 2026, de [https://aws.amazon.com/ec2/pricing/on-demand/](https://aws.amazon.com/ec2/pricing/on-demand/)
- Amazon Web Services. (2026, octubre 26). *Amazon EC2 T3 instances*. [https://aws.amazon.com/ec2/instance-types/t3/](https://aws.amazon.com/ec2/instance-types/t3/)