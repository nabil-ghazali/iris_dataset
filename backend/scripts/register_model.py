# backend/ml/register_model.py
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from pathlib import Path
import logging

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

# ------------------------ Trouver le dernier modèle local ------------------------
MODEL_DIR = Path(__file__).parent / "../model"
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
        path=str(latest_model_file),
        name=MODEL_NAME,
        description="Modèle Iris Logistic Regression",
        type="mlflow_model"  # compatible scikit-learn
    )
    registered_model = ml_client.models.create_or_update(model)
    logger.info(f"Modèle enregistré dans Azure ML : {registered_model.name}, version {registered_model.version}")

    # ------------------------ Écriture du nom du dernier modèle pour le déploiement ------------------------
    latest_model_txt = Path(__file__).parent / "../latest_model_name.txt"
    with open(latest_model_txt, "w") as f:
        f.write(registered_model.name)
    logger.info(f"Nom du dernier modèle écrit dans {latest_model_txt}")

except Exception as e:
    logger.error("Erreur lors de l'enregistrement du modèle dans Azure ML")
    logger.error(e)
    exit(1)
