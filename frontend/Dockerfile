# frontend/Dockerfile
FROM python:3.12-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le contenu du frontend dans /app
COPY . /app

# Commande pour lancer Streamlit
# Ici on indique le chemin relatif correct vers pages_predict.py
CMD ["streamlit", "run", "pages/page_predict.py", "--server.port=8501", "--server.address=0.0.0.0"]
