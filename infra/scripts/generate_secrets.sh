#!/usr/bin/env bash
# Genera una sola vez los secretos del despliegue en infra/.secrets.env (excluido de Git).
# Se niega a sobrescribir: perder la contraseña de PostgreSQL deja inaccesible la base.
set -euo pipefail

OUT="$(dirname "$0")/../.secrets.env"

if [ -e "$OUT" ]; then
  echo "Ya existe $OUT; no se sobrescribe. Bórralo a mano si de verdad quieres regenerarlo." >&2
  exit 1
fi

# Solo [A-Za-z0-9]: la contraseña de PostgreSQL va dentro de una URI y así no hay que codificarla.
alnum() { openssl rand -base64 48 | tr -dc 'A-Za-z0-9' | head -c "$1"; }

umask 077
{
  echo "# Secretos del servidor MLflow. NO entra a Git. Generado el $(date -u +%Y-%m-%dT%H:%M:%SZ)."
  echo "PG_PASSWORD=$(alnum 32)"
  echo "MLFLOW_ADMIN_PASSWORD=$(alnum 24)"
  echo "MLFLOW_FLASK_SECRET_KEY=$(openssl rand -hex 32)"
} > "$OUT"

echo "Secretos escritos en $OUT (permisos solo para tu usuario). Guárdalos en un lugar seguro."
