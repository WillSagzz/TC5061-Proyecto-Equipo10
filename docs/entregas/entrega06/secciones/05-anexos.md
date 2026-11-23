# Anexos

## Comandos para reproducir el reporte

Se ejecutan desde la raíz del repositorio, con Docker, las dependencias de `requirements.txt` instaladas y el archivo `.env` del servidor de MLflow del equipo (solo para descargar los modelos).

```bash
python -m mibici.modeling.registry download --question all   # modelos @champion a models/registry/
make compose-up      # levanta MLflow (5001) y la API (8000)
make load-test       # Locust y docker stats; resultados en reports/locust/
make compose-down    # detiene el sistema
make entrega06       # compila este PDF
```

Las gráficas y las tablas de la corrida de este reporte están en `docs/entregas/entrega06/figuras/` y `docs/entregas/entrega06/anexos/`.

## Resultados por endpoint

**Tabla 7.** Totales de la prueba de carga por endpoint (los 6 minutos, con todos los escalones).

| Endpoint | Peticiones | Errores | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|---:|---:|
| `/predict/p1` | 5,936 | 0 | 1,200 | 5,100 | 6,900 |
| `/predict/p2` | 2,935 | 0 | 3,300 | 7,900 | 9,600 |
| `/predict/p3` | 1,034 | 0 | 540 | 4,900 | 6,900 |
| `/health` | 938 | 0 | 630 | 4,800 | 6,200 |
| Total | 10,843 | 0 | 1,400 | 6,300 | 8,200 |

## Prueba de configuraciones de la API

Las cuatro configuraciones de la Tabla 5 se corrieron sobre el mismo sistema de Compose. Los *workers* se cambian con la variable `API_WORKERS`. El límite de hilos se agregó al servicio `api` con un archivo adicional de Compose:

```yaml
# hilos.yml
services:
  api:
    environment:
      OMP_NUM_THREADS: "1"
```

```bash
API_WORKERS=2 docker compose -f docker-compose.yml -f hilos.yml up -d api
```

En cada configuración Locust corrió 90 segundos con 100 usuarios fijos del mismo escenario (`StationOperator`), sin la carga escalonada. Los resultados están en `anexos/prueba_configuraciones.csv`.

## Precios de referencia

**Tabla 8.** Instancias de 2 vCPU y 4 GB bajo demanda, con Linux, y el costo por millón de predicciones con la capacidad medida en la prueba.

| Proveedor | Instancia | USD al mes | USD por millón, saturada | USD por millón, demanda de MiBici |
|---|---|---:|---:|---:|
| AWS | EC2 `t3.medium`, `us-east-1` | 30.37 | 0.34 | 76 |
| Azure | `Standard_B2s`, East US | 30.37 | 0.34 | 76 |
| DigitalOcean | Basic Droplet de 4 GB | 24.00 | 0.27 | 60 |

## Referencias

- Amazon Web Services. (s.f.-a). *Amazon EC2 On-Demand pricing*. Recuperado el 22 de noviembre de 2026, de <https://aws.amazon.com/ec2/pricing/on-demand/>
- Amazon Web Services. (s.f.-b). *Amazon EC2 T3 instances*. Recuperado el 22 de noviembre de 2026, de <https://aws.amazon.com/ec2/instance-types/t3/>
- CloudPrice. (s.f.). *Standard_B2s specs and pricing*. Recuperado el 22 de noviembre de 2026, de <https://cloudprice.net/vm/Standard_B2s>
- DigitalOcean. (s.f.). *Droplet pricing*. Recuperado el 22 de noviembre de 2026, de <https://www.digitalocean.com/pricing/droplets>
- FinOps Foundation. (s.f.). *FinOps for AI overview*. Recuperado el 22 de noviembre de 2026, de <https://www.finops.org/wg/finops-for-ai-overview/>
