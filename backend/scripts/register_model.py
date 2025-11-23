import os
import json
import logging
from pathlib import Path
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

# ------------------------ Logger ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------ Charger config.json ------------------------
config_dir = os.environ.get("AZUREML_CONFIG_DIR")

if config_dir is None:
    logger.error("Variable d'environnement AZUREML_CONFIG_DIR non définie.")
    exit(1)

config_path = Path(config_dir) / "config.json"

if not config_path.exists():
    logger.error(f"Fichier config.json non trouvé : {config_path}")
    exit(1)

logger.info(f"Chargement de la configuration Azure ML depuis : {config_path}")

with open(config_path, "r") as f:
    config = json.load(f)

try:
    subscription_id = config["subscription_id"]
    resource_group = config["resource_group"]
    workspace = config["workspace_name"]

except KeyError as e:
    logger.error(f"Clé manquante dans config.json : {e}")
    exit(1)

# ------------------------ Authentification ------------------------
# Tu peux aussi utiliser DefaultAzureCredential() si variables d'env définies
credential = ClientSecretCredential(
    tenant_id=os.environ.get("AZURE_TENANT_ID"),
    client_id=os.environ.get("AZURE_CLIENT_ID"),
    client_secret=os.environ.get("AZURE_CLIENT_SECRET")
)

ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace
)

logger.info("Connecté à Azure ML via SDK v2.")

# ------------------------ Trouver le dernier modèle local ------------------------
MODEL_DIR = Path(__file__).parent.parent / "model"
model_files = list(MODEL_DIR.glob("model_*.pkl"))

if not model_files:
    logger.error(f"Aucun modèle trouvé dans {MODEL_DIR}")
    exit(1)

latest_model_file = max(model_files, key=lambda f: f.stat().st_mtime)
logger.info(f"Dernier modèle trouvé : {latest_model_file.name}")

# ------------------------ Enregistrement dans Azure ML ------------------------
MODEL_NAME = "iris_model"

try:
    model = Model(
        name=MODEL_NAME,
        path=str(latest_model_file),
        type="custom_model",
        description="Modèle Iris Logistic Regression"
    )

    registered_model = ml_client.models.create_or_update(model)

    logger.info(f"Modèle enregistré : {registered_model.name}, version {registered_model.version}")

    # Sauvegarder le nom pour le déploiement
    latest_model_txt = Path(__file__).parent / "../latest_model_name.txt"
    with open(latest_model_txt, "w") as f:
        f.write(registered_model.name)

    logger.info(f"Nom du modèle enregistré écrit dans {latest_model_txt}")

except Exception as e:
    logger.error("Erreur durant l'enregistrement du modèle")
    logger.error(e)
    exit(1)
