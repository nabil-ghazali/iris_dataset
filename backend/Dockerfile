FROM python:3.12-slim

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# copier requirements
COPY requirements.txt .

# installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# copier le code
COPY . .

# Exposer le port FastAPI
EXPOSE 8000

# lancer l’API
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
