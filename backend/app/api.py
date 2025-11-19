from fastapi import FastAPI
import joblib
from pathlib import Path
from pydantic import BaseModel
import logging
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------ Configuration logg ---------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------- Modèle de requête pour /predict ----------------------------
# IrisData est le schéma Pydantic pour valider les entrées JSON POST /predict. Et convertis les types quand il peut
class IrisData(BaseModel):
    """Modèle de données pour prédiction Iris

    Attributes:
        sepal_length (float): Longueur du sépale en cm
        sepal_width (float): Largeur du sépale en cm
        petal_length (float): Longueur du pétale en cm
        petal_width (float): Largeur du pétale en cm
    """

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Dictionnaire de correspondance classes → noms
IRIS_CLASS = {0: "setosa", 1: "versicolor", 2: "virginica"}


# -------------------------------- création de l'application --------------------------------
app = FastAPI(title="Iris predict")


# ------------------------ Autoriser l'accès depuis le frontend (Streamlit)-------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------- Chargement du dernier model disponible --------------------------
MODEL_DIR = Path(__file__).parent.parent / "model"
# récupère le dernier modèle sauvegardé
model_files = list(MODEL_DIR.glob("model_*.pkl"))
if not model_files:
    raise FileNotFoundError(f"Aucun modèle trouvé dans {MODEL_DIR}")
latest_model_file = max(model_files, key=lambda f: f.stat().st_mtime)
model = joblib.load(latest_model_file)
logger.info(f"Modèle chargé : {latest_model_file.name}")


# --------------------------------- création des routes --------------------------------------


# Route GET simple
@app.get("/")
def root():
    """Endpoint racine pour vérifier que l'API fonctionne

    Returns:
        dict: Message de bienvenue
    """
    logger.info("GET / appelé")
    return {"message": "Bienvenue dans le projet Iris Predict"}


# Route POST /predict
@app.post("/predict")
def predict(data: IrisData):
    """Prédit la variété d'une fleur Iris à partir de ses mesures.

    Args:
        data (IrisData): Mesures de la fleur (sépale et pétale)

    Returns:
        Dict[str, Union[str, int]]: Clé 'prediction' -> code entier, clé 'variety' -> nom de la variété
    """
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
