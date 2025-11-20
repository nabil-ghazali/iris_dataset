# score.py
import joblib
import os
import numpy as np
from pathlib import Path
import logging

# ------------------------ Logger ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------ Initialisation du modèle ------------------------
model = None
MODEL_DIR = Path(__file__).parent.parent / "model"  # ../model
latest_model_file = None

def init():
    global model, latest_model_file
    # Récupérer le dernier modèle sauvegardé
    model_files = list(MODEL_DIR.glob("model_*.pkl"))
    if not model_files:
        logger.error(f"Aucun modèle trouvé dans {MODEL_DIR}")
        raise FileNotFoundError(f"Aucun modèle trouvé dans {MODEL_DIR}")
    
    latest_model_file = max(model_files, key=lambda f: f.stat().st_mtime)
    model = joblib.load(latest_model_file)
    logger.info(f"Modèle chargé pour l'inférence : {latest_model_file.name}")


# ------------------------ Fonction run pour Azure ML ------------------------
def run(input_data):
    """
    input_data: dict avec clé 'data' -> liste de listes de features
    Exemple: {"data": [[5.1, 3.5, 1.4, 0.2]]}
    """
    global model
    try:
        X = np.array(input_data["data"])
        preds = model.predict(X)
        return {"predictions": preds.tolist()}
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        return {"error": str(e)}


# ------------------------ Test local ------------------------
if __name__ == "__main__":
    init()
    sample = {"data": [[5.1, 3.5, 1.4, 0.2]]}
    result = run(sample)
    print("Test prédiction :", result)
