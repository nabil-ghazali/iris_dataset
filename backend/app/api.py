from fastapi import FastAPI
import joblib
from pathlib import Path
from pydantic import BaseModel
import logging

# -------------- Configuration logg ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------- Modèle de requête pour /predict ----------------
# IrisData est le schéma Pydantic pour valider les entrées JSON POST /predict. Et convertis les types quand il peut
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Dictionnaire de correspondance classes → noms
IRIS_CLASS = {0: "setosa", 1: "versicolor", 2: "virginica"}


# --------------- création de l'application ----------------------
app = FastAPI(title="Iris predict")

# --------------- Chargement du dernier model disponible ---------
MODEL_DIR = Path(__file__).parent.parent / "model"
# récupère le dernier modèle sauvegardé
model_files = list(MODEL_DIR.glob("model_*.pkl"))
if not model_files:
    raise FileNotFoundError(f"Aucun modèle trouvé dans {MODEL_DIR}")
latest_model_file = max(model_files, key=lambda f: f.stat().st_mtime)
model = joblib.load(latest_model_file)
logger.info(f"Modèle chargé : {latest_model_file.name}")

# --------------- création des routes ------------------------


# Route GET simple
@app.get("/")
def root():
    logger.info("GET / appelé")
    return {"message": "Bienvenue dans le projet Iris Predict"}


# Route POST /predict
@app.post("/predict")
def predict(data: IrisData):
    logger.info(f"POST /predict appelé avec data={data}")
    X = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]

    try:
        # Le modèle prédit un entier (0/1/2)
        pred = model.predict(X)[0]
        # Convertir en nom de variété
        variety = IRIS_CLASS.get(int(pred), "Unknown")
        logger.info(f"Prediction : {pred} ({variety})")
        return {"prediction": int(pred), "variety": variety}
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        return {"error": str(e)}
