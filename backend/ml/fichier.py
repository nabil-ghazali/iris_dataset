import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow.sklearn
import shutil
import os
import logging

# -------------------- Configuration du logger --------------------
"""
logging.basicConfig → configure le logger global pour ton script.

level=logging.INFO → on capture tous les messages de niveau INFO et supérieurs (INFO, WARNING, ERROR, CRITICAL).

Astuce : DEBUG < INFO < WARNING < ERROR < CRITICAL

format="..." → définit l'apparence des messages :

%(asctime)s → affiche la date et l'heure

%(levelname)s → affiche le niveau (INFO, ERROR, etc.)

%(message)s → affiche le texte du message

handlers=[logging.StreamHandler()] → indique où envoyer les messages. Ici, c'est la console (par défaut sys.stdout).
"""
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


# Dossier où enregistrer tous les modèles
MODEL_DIR = "../model"  # backend/model depuis le dossier backend/ml
os.makedirs(MODEL_DIR, exist_ok=True)

# Création de l'experiment MLflow
mlflow.set_experiment("Mlflow Iris")

# Chargement du dataset
logger.info("Chargement du dataset Iris")
X, y = datasets.load_iris(return_X_y=True)

# Division train/test
logger.info("Séparation train/test")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparamètres
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "multinomial",  # pour éviter le warning futur
    "random_state": 8888,
}

# Activation de l'autologging
mlflow.sklearn.autolog()  # MLflow autolog → sauvegarde du modèle dans mlruns/.../artifacts/model/

# ----------------------------   Démarrage du run  -----------------------
# run contient toutes les infos sur le run actif, notamment run.info.run_id.
# Au lieu de mettre <run_id> à la main, MLflow fournit mlflow.active_run() :
# active_run = mlflow.active_run()
# run_id = active_run.info.run_id
# Cela te donne automatiquement l’ID du run courant.
# Plus besoin de chercher dans mlruns/0/<run_id> manuellement.
logger.info("Démarrage du run MLflow")
with mlflow.start_run() as run:
    # Entrainement du modèle
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Evaluation du modèle
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    logger.info(f"Metrics: Accuracy={acc:.4f}, F1-score={f1:.4f}")

    # Récupération automatique du chemin local du modèle MLflow
    model_artifact_path = "model"  # chemin relatif dans les artifacts
    model_local_path = mlflow.artifacts.download_artifacts(  # mlflow.artifacts.download_artifacts() → récupère le model.pkl localement
        artifact_path=model_artifact_path, run_id=run.info.run_id
    )

    # Chemin destination final
    destination = os.path.join(MODEL_DIR, f"model_{run.info.run_id}.pkl")

    # Copie du model.pkl
    shutil.copy2(os.path.join(model_local_path, "model.pkl"), destination)
    logger.info(f"Modèle sauvegardé dans {destination}")
