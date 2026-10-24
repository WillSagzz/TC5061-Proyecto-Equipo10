#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TC5061-Proyecto-Equipo10
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

# Servidor MLflow compartido en Azure (infra/). Solo lo usa quien administra la suscripción.
AZ_RG = rg-mibici-equipo10
AZ_LOCATION = mexicocentral
AZ_APP = mlflow
INFRA_SECRETS = infra/.secrets.env
# Preámbulo común de los objetivos infra-*: exige los secretos y los exporta al entorno
INFRA_ENV = test -f $(INFRA_SECRETS) || { echo "Falta $(INFRA_SECRETS): corre make infra-secrets"; exit 1; }; set -a && . ./$(INFRA_SECRETS) && set +a
INFRA_URL = https://$$(az containerapp show -g $(AZ_RG) -n $(AZ_APP) --query properties.configuration.ingress.fqdn -o tsv)

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Rebuild the data version described in datapackage.json: download, verify hashes, clean, write Parquet
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mibici.dataset


## Freeze a new data version and write datapackage.json (optional: VERSION=2026-10 START=... END=...)
.PHONY: snapshot
snapshot:
	$(PYTHON_INTERPRETER) -m mibici.datapackage snapshot $(if $(VERSION),--version $(VERSION)) $(if $(START),--start $(START)) $(if $(END),--end $(END))


## Validate the local data against datapackage.json (hashes, sizes and every row of each schema)
.PHONY: validate
validate:
	$(PYTHON_INTERPRETER) -m mibici.datapackage validate


## Train the baselines of P1, P2 and P3 and log them to MLflow
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m mibici.modeling.train all


## Compile Entrega 01 (Markdown -> HTML -> PDF) into docs/entregas/entrega01/build/
.PHONY: entrega01
entrega01:
	$(PYTHON_INTERPRETER) docs/entregas/build.py docs/entregas/entrega01


#################################################################################
# SERVIDOR MLFLOW COMPARTIDO EN AZURE (infra/README.md)                         #
#################################################################################
# Requiere Azure CLI con sesión iniciada (`az login`) en la suscripción del equipo.
# Los secretos viven en infra/.secrets.env, fuera de Git.

## Install Bicep, the Container Apps extension and register the Azure providers (once)
.PHONY: infra-prereqs
infra-prereqs:
	az bicep install
	az extension add --name containerapp --upgrade --allow-preview true
	az provider register --namespace Microsoft.App --wait
	az provider register --namespace Microsoft.DBforPostgreSQL --wait
	az provider register --namespace Microsoft.Storage --wait
	az provider register --namespace Microsoft.OperationalInsights --wait
	az provider register --namespace Microsoft.ManagedIdentity --wait

## Generate the deployment secrets once into infra/.secrets.env (never versioned)
.PHONY: infra-secrets
infra-secrets:
	bash infra/scripts/generate_secrets.sh

## Preview the changes Bicep would make (what-if), without deploying
.PHONY: infra-plan
infra-plan:
	az group create --name $(AZ_RG) --location $(AZ_LOCATION) --output none
	$(INFRA_ENV) && \
	export AZ_ADMIN_OBJECT_ID=$$(az ad signed-in-user show --query id -o tsv 2>/dev/null || true) && \
	az deployment group what-if --resource-group $(AZ_RG) --name $(AZ_APP) \
		--template-file infra/main.bicep --parameters infra/main.bicepparam

## Deploy or update the MLflow server (resource group + Bicep) and print its URL
.PHONY: infra-deploy
infra-deploy:
	az group create --name $(AZ_RG) --location $(AZ_LOCATION) --output none
	$(INFRA_ENV) && \
	export AZ_ADMIN_OBJECT_ID=$$(az ad signed-in-user show --query id -o tsv 2>/dev/null || true) && \
	az deployment group create --resource-group $(AZ_RG) --name $(AZ_APP) \
		--template-file infra/main.bicep --parameters infra/main.bicepparam \
		--query "properties.outputs.mlflowUrl.value" -o tsv

## Print the URL of the shared MLflow server
.PHONY: infra-url
infra-url:
	@echo $(INFRA_URL)

## Create the users of infra/users.txt (local, not versioned) and assign their role (PRUNE=1 deletes others)
.PHONY: infra-users
infra-users:
	@test -f infra/users.txt || { echo "Falta infra/users.txt: copia infra/users.example.txt y pon los usuarios del equipo"; exit 1; }
	$(INFRA_ENV) && export MLFLOW_URL=$(INFRA_URL) && \
	$(PYTHON_INTERPRETER) infra/scripts/create_users.py $(if $(filter-out 0 false no,$(PRUNE)),--prune,)

## Reset a user's password as admin (USUARIO=nombre); the new one goes to infra/.users.env
.PHONY: infra-reset-password
infra-reset-password:
	@test -n "$(USUARIO)" || { echo "Uso: make infra-reset-password USUARIO=nombre"; exit 1; }
	$(INFRA_ENV) && export MLFLOW_URL=$(INFRA_URL) && \
	$(PYTHON_INTERPRETER) infra/scripts/create_users.py --reset $(USUARIO)

## Smoke test of the shared server with the credentials of your .env
.PHONY: infra-check
infra-check:
	$(PYTHON_INTERPRETER) infra/scripts/check_server.py

## Follow the server logs
.PHONY: infra-logs
infra-logs:
	az containerapp logs show -g $(AZ_RG) -n $(AZ_APP) --type console --follow --tail 100

## Delete the whole resource group (asks for confirmation)
.PHONY: infra-destroy
infra-destroy:
	az group delete --name $(AZ_RG)

## Change your own password on the shared MLflow server (uses .env; no Azure needed)
.PHONY: mlflow-password
mlflow-password:
	$(PYTHON_INTERPRETER) infra/scripts/change_password.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
