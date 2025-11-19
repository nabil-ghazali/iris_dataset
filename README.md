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



Explications étape par étape
Étape	                              Explication
Checkout repository	              Récupère tout le code du repo sur le runner GitHub (Ubuntu).

Setup Python	                      Configure la version Python dont tu as besoin (ici 3.12).

Install backend dependencies	      Installe toutes les librairies nécessaires au backend pour que les tests     
                                      fonctionnent.

Run backend tests	              Pytest s’exécute ici pour s’assurer que ton code fonctionne avant de     
                                      déployer quoi que ce soit.

Deploy documentation	              Installe MkDocs et déploie la doc sur GitHub Pages via mkdocs gh-deploy.

Login DockerHub	                      Se connecte à DockerHub en utilisant les secrets pour pouvoir push les images.

Build & push images backend/frontend  Construit les images Docker et les push sur DockerHub pour que tu puisses les 
                                      utiliser pour le déploiement (prod ou staging).



       +-------------------+
       |    GitHub Repo    |
       | (backend + front) |
       +-------------------+
                |
                | Push / Merge (branche main)
                v
       +-------------------+
       | GitHub Actions CI |
       |                   |
       | 1. Installer deps |
       | 2. Lancer tests   |
       | 3. Déployer docs  |
       | 4. Build Docker   |
       | 5. Push Docker    |
       +-------------------+
           |                |
           | Docker images  | 
           v                v
+-------------------+   +-------------------+
| DockerHub Backend |   | DockerHub Frontend|
+-------------------+   +-------------------+
           |                |
           | docker pull     | docker pull
           v                v
   +-------------------+   +-------------------+
   | Backend Container |   | Frontend Container|
   |   (FastAPI)       |   |  (Streamlit)     |
   +-------------------+   +-------------------+
           |                ^
           | API calls      | UI requests
           +----------------+

Explication des flux

1. GitHub Repository

        - Contient le code frontend (frontend/) et backend (backend/) ainsi que docker-compose.yml et la documentation.

        - Tout push ou merge sur la branche main déclenche le pipeline GitHub Actions.

2. GitHub Actions (CI/CD)

        - Étape 1 : Installe les dépendances Python pour backend et frontend.

        - Étape 2 : Lance les tests (pytest pour le backend).

        - Étape 3 : Déploie la documentation (par exemple via mkdocs gh-deploy).

        - Étape 4 : Build des images Docker pour backend et frontend.

        - Étape 5 : Push des images Docker sur DockerHub.

3. DockerHub

        - Stocke les images Docker construites pour le backend et frontend.

        - Sert de registry pour que ton infrastructure ou ton serveur de production puisse récupérer les images.

4. Containers backend et frontend

        - Backend (FastAPI) tourne sur un port (ex: 8000).

        - Frontend (Streamlit) tourne sur un autre port (ex: 8501) et communique avec le backend via des appels API.

        - Le frontend récupère les données du modèle en appelant le backend /predict.

5. Flux de communication

        - Utilisateur interagit avec le frontend via navigateur.

        - Frontend envoie les données vers backend via une requête POST /predict.

        - Backend renvoie la prédiction.# Trigger workflow
