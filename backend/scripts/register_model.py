# backend/ml/register_model.py
import os
from pathlib import Path
import logging
from azureml.core import Workspace, Model

# ------------------------ Logger ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------ Connexion au Workspace ------------------------
try:
    logger.info("Connexion au workspace Azure ML...")
    ws = Workspace.from_config()  # lit le fichier config.json
    logger.info(f"Connecté au workspace : {ws.name}")
except Exception as e:
    logger.error("Impossible de se connecter au workspace Azure ML")
    logger.error(e)
    exit(1)

# ------------------------ Récupération du dernier modèle local ------------------------
MODEL_DIR = (Path(__file__).parent / "../model").resolve()
model_files = list(MODEL_DIR.glob("model_*.pkl"))

if not model_files:
    logger.error(f"Aucun modèle trouvé dans {MODEL_DIR}")
    exit(1)

latest_model_file = max(model_files, key=lambda f: f.stat().st_mtime)
logger.info(f"Dernier modèle trouvé : {latest_model_file.name}")

# ------------------------ Enregistrement dans Azure ML ------------------------
MODEL_NAME = "iris_model"  # nom officiel dans Azure ML
try:
    model = Model.register(
        workspace=ws,
        model_path=str(latest_model_file),  # chemin vers le fichier .pkl
        model_name=MODEL_NAME,
        tags={"run_type": "CI/CD pipeline"},
        description="Modèle Iris enregistré via GitHub Actions"
    )
    logger.info(f"Modèle enregistré dans Azure ML : {model.name}, version {model.version}")

    # ------------------------ Écriture du fichier pour deploy_endpoint.py ------------------------
    latest_model_file_txt = Path(__file__).parent / "latest_model_name.txt"
    with open(latest_model_file_txt, "w") as f:
        f.write(model.name)
    logger.info(f"Nom du modèle écrit dans {latest_model_file_txt}")

except Exception as e:
    logger.error("Erreur lors de l'enregistrement du modèle dans Azure ML")
    logger.error(e)
    exit(1)
