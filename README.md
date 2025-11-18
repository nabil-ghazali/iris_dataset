# iris_dataset
Flux complet de bout en bout :

- fichier.py → Entraîne le modèle → sauvegarde dans backend/model/model_<run_id>.pkl.

- api.py → Charge le dernier modèle sauvegardé → FastAPI expose /predict.

- pages_predict.py → L’utilisateur saisit les valeurs → Streamlit envoie au backend → FastAPI prédit →

Visualisation du flux :
Utilisateur (Streamlit) 
        │
        │ POST JSON (inputs)
        ▼
FastAPI /predict (api.py)
        │
        │ Charge le dernier model.pkl
        │ Predict -> renvoie JSON
        ▼
Utilisateur (Streamlit) affiche prediction
