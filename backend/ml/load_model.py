# load_model.py
import os
import joblib
import numpy as np
from azureml.core import Workspace, Model
from sklearn.datasets import load_iris

# ---------------------- CONFIGURATION ----------------------
# Le chemin de ton fichier de config Azure ML téléchargé
CONFIG_FILE = "config.json"  # mettre le fichier config téléchargé à la racine

# Nom du modèle enregistré dans Azure ML
MODEL_NAME = "iris_model"  # adapte selon le nom que tu as utilisé

# Dossier local pour télécharger le modèle
LOCAL_MODEL_DIR = "./model"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# ---------------------- CONNEXION AU WORKSPACE ----------------------
print("Connexion au Workspace Azure ML...")
ws = Workspace.from_config(path=CONFIG_FILE)
print(f"Connecté au workspace : {ws.name}")

# ---------------------- RÉCUPÉRATION DU MODÈLE ----------------------
print(f"Téléchargement du modèle '{MODEL_NAME}'...")
model = Model(ws, name=MODEL_NAME)

# Télécharger le modèle localement
model_path = model.download(target_dir=LOCAL_MODEL_DIR, exist_ok=True)
print(f"Modèle téléchargé dans : {model_path}")

# ---------------------- TEST DE PRÉDICTION ----------------------
print("Chargement du modèle et test de prédiction...")
# On suppose que c'est un modèle scikit-learn enregistré via joblib
loaded_model = joblib.load(os.path.join(model_path))

# Exemple avec les 5 premières fleurs du dataset Iris
X, y = load_iris(return_X_y=True)
sample_X = X[:5]

predictions = loaded_model.predict(sample_X)
print(f"Prédictions pour les 5 premiers échantillons : {predictions}")
print(f"Valeurs réelles : {y[:5]}")
