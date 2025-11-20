# from fastapi import FastAPI
# import joblib
# from pathlib import Path
# from pydantic import BaseModel
# import logging
# from fastapi.middleware.cors import CORSMiddleware

# # ------------------------------ Configuration logg ---------------------------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # ----------------------------- Modèle de requête pour /predict ----------------------------
# # IrisData est le schéma Pydantic pour valider les entrées JSON POST /predict. Et convertis les types quand il peut
# class IrisData(BaseModel):
#     """Modèle de données pour prédiction Iris

#     Attributes:
#         sepal_length (float): Longueur du sépale en cm
#         sepal_width (float): Largeur du sépale en cm
#         petal_length (float): Longueur du pétale en cm
#         petal_width (float): Largeur du pétale en cm
#     """

#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float


# # Dictionnaire de correspondance classes → noms
# IRIS_CLASS = {0: "setosa", 1: "versicolor", 2: "virginica"}


# # -------------------------------- création de l'application --------------------------------
# app = FastAPI(title="Iris predict")


# # ------------------------ Autoriser l'accès depuis le frontend (Streamlit)-------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # ou ["http://localhost:8501"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # -------------------------- Chargement du dernier model disponible --------------------------
# MODEL_DIR = Path(__file__).parent.parent / "model"
# # récupère le dernier modèle sauvegardé
# model_files = list(MODEL_DIR.glob("model_*.pkl"))
# if not model_files:
#     raise FileNotFoundError(f"Aucun modèle trouvé dans {MODEL_DIR}")
# latest_model_file = max(model_files, key=lambda f: f.stat().st_mtime)
# model = joblib.load(latest_model_file)
# logger.info(f"Modèle chargé : {latest_model_file.name}")


# # --------------------------------- création des routes --------------------------------------


# # Route GET simple
# @app.get("/")
# def root():
#     """Endpoint racine pour vérifier que l'API fonctionne

#     Returns:
#         dict: Message de bienvenue
#     """
#     logger.info("GET / appelé")
#     return {"message": "Bienvenue dans le projet Iris Predict"}


# # Route POST /predict
# @app.post("/predict")
# def predict(data: IrisData):
#     """Prédit la variété d'une fleur Iris à partir de ses mesures.

#     Args:
#         data (IrisData): Mesures de la fleur (sépale et pétale)

#     Returns:
#         Dict[str, Union[str, int]]: Clé 'prediction' -> code entier, clé 'variety' -> nom de la variété
#     """
#     logger.info(f"POST /predict appelé avec data={data}")
#     X = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]

#     try:
#         # Le modèle prédit un entier (0/1/2)
#         pred = model.predict(X)[0]
#         # Convertir en nom de variété
#         variety = IRIS_CLASS.get(int(pred), "Unknown")
#         logger.info(f"Prediction : {pred} ({variety})")
#         return {"prediction": int(pred), "variety": variety}
#     except Exception as e:
#         logger.error(f"Erreur lors de la prédiction : {e}")
#         return {"error": str(e)}

from fastapi import FastAPI
from pydantic import BaseModel
import logging
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import os
from azureml.core import Workspace, Model

# ------------------------------ Logger ---------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------- Modèle de requête -----------------------------
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

IRIS_CLASS = {0: "setosa", 1: "versicolor", 2: "virginica"}

# ------------------------ Création de l'application -------------------------
app = FastAPI(title="Iris predict")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------- Répertoire des modèles --------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_NAME = "iris_model"

# -------------------------- Fonction pour charger le modèle -------------------
def load_model():
    """Charge le modèle Azure ML, sinon fallback local"""
    try:
        logger.info("Tentative de connexion au Workspace Azure ML...")
        ws = Workspace.from_config()
        logger.info(f"Connecté au workspace : {ws.name}")

        model = Model(ws, name=MODEL_NAME)
        model_path = model.download(target_dir=MODEL_DIR, exist_ok=True)
        logger.info(f"Modèle Azure ML téléchargé dans : {model_path}")

        loaded_model = joblib.load(model_path)
        logger.info("Modèle Azure ML chargé avec succès !")
        return loaded_model

    except Exception as e:
        logger.warning(f"Impossible de charger le modèle depuis Azure ML : {e}")
        logger.info("Tentative de charger le dernier modèle local disponible...")

        # Récupérer le dernier modèle local
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
        if not model_files:
            raise FileNotFoundError(f"Aucun modèle trouvé dans {MODEL_DIR}")

        latest_model_file = max(model_files, key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)))
        model_local_path = os.path.join(MODEL_DIR, latest_model_file)
        loaded_model = joblib.load(model_local_path)
        logger.info(f"Modèle local chargé : {latest_model_file}")
        return loaded_model

# Charger le modèle au démarrage
model = load_model()

# -------------------------- Routes API --------------------------------------
@app.get("/")
def root():
    logger.info("GET / appelé")
    return {"message": "Bienvenue dans le projet Iris Predict"}

@app.post("/predict")
def predict(data: IrisData):
    logger.info(f"POST /predict appelé avec data={data}")
    X = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])

    try:
        pred = model.predict(X)[0]
        variety = IRIS_CLASS.get(int(pred), "Unknown")
        logger.info(f"Prediction : {pred} ({variety})")
        return {"prediction": int(pred), "variety": variety}
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        return {"error": str(e)}
