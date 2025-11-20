import os
import joblib
from azureml.core import Workspace, Model
import logging
"""
Ce fichier contient la logique pour récupérer le dernier modèle depuis Azure ML et le charger en mémoire.

"""
# ------------------------ Logger ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------ Variables --------------------------
MODEL_NAME = "iris_model"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------ Fonctions --------------------------
def load_model_from_azure():
    try:
        # Connexion au workspace Azure ML
        logger.info("Connexion au Workspace Azure ML...")
        ws = Workspace.from_config()
        logger.info(f"Connecté au workspace : {ws.name}")

        # Récupération du modèle
        logger.info(f"Téléchargement du modèle '{MODEL_NAME}'...")
        model = Model(ws, name=MODEL_NAME)
        model_path = model.download(target_dir=MODEL_DIR, exist_ok=True)
        logger.info(f"Modèle téléchargé dans : {model_path}")

        # Chargement du modèle avec joblib
        loaded_model = joblib.load(model_path)
        logger.info("Modèle chargé avec succès !")
        return loaded_model

    except Exception as e:
        logger.error("Erreur lors du chargement du modèle depuis Azure ML : %s", e)
        return None

# ------------------------ Chargement --------------------------
model = load_model_from_azure()
