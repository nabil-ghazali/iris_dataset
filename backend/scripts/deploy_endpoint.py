# backend/ml/deploy_endpoint.py
import os
import time
import logging
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration

# ------------------------ Logger ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------ Azure ML Configuration ------------------------
try:
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.environ["AZURE_RESOURCE_GROUP"]
    workspace = os.environ["AZURE_ML_WORKSPACE"]
except KeyError as e:
    logger.error(f"Variable d'environnement manquante : {e}")
    exit(1)

credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace)
logger.info("Connecté à Azure ML via SDK v2.")

# ------------------------ Lecture du dernier modèle enregistré ------------------------
MODEL_INFO_FILE = Path(__file__).parent / "../latest_model_name.txt"
if not MODEL_INFO_FILE.exists():
    logger.error("latest_model_name.txt introuvable. Le modèle n'a pas été enregistré.")
    exit(1)

with open(MODEL_INFO_FILE, "r") as f:
    model_name = f.read().strip()

logger.info(f"Modèle récupéré pour déploiement : {model_name}")

# ------------------------ Endpoint ------------------------
endpoint_name = "iris-endpoint"

# Création ou récupération de l'endpoint
try:
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    logger.info(f"Endpoint existant trouvé : {endpoint_name}")
except Exception:
    logger.info(f"Endpoint inexistant → création de {endpoint_name}")
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode="key"
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info(f"Endpoint créé : {endpoint_name}")

# ------------------------ Configuration du scoring script ------------------------
code_config = CodeConfiguration(
    code=str(Path(__file__).parent),  # dossier contenant score.py
    scoring_script="score.py"
)

# ------------------------ Déploiement ------------------------
deployment_name = f"deploy-{int(time.time())}"
deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=endpoint_name,
    model=model_name,
    code_configuration=code_config,
    instance_type="Standard_DS1_v2",
    instance_count=1,
)

logger.info(f"Lancement du déploiement {deployment_name}...")
ml_client.online_deployments.begin_create_or_update(deployment).result()
logger.info(f"Deployment créé : {deployment_name}")

# ------------------------ Définir ce deployment par défaut ------------------------
ml_client.online_endpoints.begin_update(
    ManagedOnlineEndpoint(
        name=endpoint_name,
        default_deployment=deployment_name
    )
).result()

logger.info(f" Modèle déployé sur l'endpoint : {endpoint_name}")
logger.info(f" Deployment actif : {deployment_name}")
