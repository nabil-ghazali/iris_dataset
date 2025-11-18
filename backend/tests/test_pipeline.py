import pytest
import joblib  # joblib = “boîte magique pour mettre ton modèle dedans et le ressortir quand tu veux”
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "model"
# P ath(__file__)           # transforme le chemin du fichier actuel en objet Path
# .parent                  # prend le dossier contenant ce fichier
# .parent                  # remonte encore d’un niveau dans l’arborescence
# / "model"                # ajoute le dossier "model" à ce chemin


# Vérifie que le modèle existe dans le dossier model/.
# Vérifie que le modèle a bien la méthode .predict().
def test_load_latest_model():
    model_files = list(
        MODEL_DIR.glob("model_*.pkl")
    )  # → cherche tous les fichiers dans MODEL_DIR dont le nom commence par model_ et finit par .pkl.
    assert len(model_files) > 0, (
        "Aucun modèle trouvé"
    )  # Vérifie qu’il y a au moins un fichier correspondant. Sinon leve une erreur
    latest_model_file = max(  # max() → cherche l’élément “le plus grand” selon une clé donnée.
        model_files,
        key=lambda f: f.stat().st_mtime,  # f.stat() → donne des infos sur le fichier (taille, date, etc.)  # trouve le dernier fichier modifié
    )
    model = joblib.load(  # → ouvre un fichier .joblib ou .pkl et récupère l’objet Python qu’il contient.
        latest_model_file  # le chemin vers le fichier du modèle le plus récent
    )

    # Test que le modèle possède la méthode predict
    assert hasattr(model, "predict")  # on vérifie si model a une méthode predict.
    # Avec Joblib, tu peux charger n’importe quel fichier, mais rien ne garantit qu’il contient un vrai modèle.
    # Cette assertion permet de détecter les erreurs tôt


# Teste que le modèle prédit une valeur valide pour un input donné.
def test_model_prediction_shape():
    import numpy as np

    latest_model_file = max(
        list(MODEL_DIR.glob("model_*.pkl")), key=lambda f: f.stat().st_mtime
    )
    model = joblib.load(latest_model_file)
    X_test = np.array([[5.1, 3.5, 1.4, 0.2]])
    pred = model.predict(X_test)
    assert (
        pred.shape[0] == 1
    )  # Vérifie que le modèle renvoie exactement une prédiction.
    assert (
        pred[0] in [0, 1, 2]
    )  # Vérifie que la valeur prédite est parmi les classes attendues (0, 1 ou 2 pour iris dataset).
