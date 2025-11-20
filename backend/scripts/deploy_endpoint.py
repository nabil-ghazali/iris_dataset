# backend/ml/deploy_endpoint.py
import os
import time
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration

# -----------------------------
# Configuration Azure ML
# -----------------------------
subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group = os.environ["AZURE_RESOURCE_GROUP"]
workspace = os.environ["AZURE_ML_WORKSPACE"]

credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace)

# -----------------------------
# Récupération du dernier modèle enregistré
# -----------------------------
MODEL_INFO_FILE = "latest_model_name.txt"
if not os.path.exists(MODEL_INFO_FILE):
    raise FileNotFoundError("latest_model_name.txt introuvable. Le modèle n'a pas été enregistré.")

with open(MODEL_INFO_FILE, "r") as f:
    model_name = f.read().strip()

print(f"➡ Modèle récupéré pour déploiement : {model_name}")

# -----------------------------
# Nom de l'endpoint
# -----------------------------
endpoint_name = "iris-endpoint"

# -----------------------------
# Créer ou mettre à jour l'endpoint
# -----------------------------
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    auth_mode="key"  # authentification via clé
)

try:
    ml_client.online_endpoints.get(name=endpoint_name)
    print("✓ Endpoint déjà existant, mise à jour...")
except Exception:
    print("ℹ Endpoint inexistant → création...")
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# -----------------------------
# Configuration du code de scoring
# -----------------------------
code_config = CodeConfiguration(
    code="./backend/scripts",   # dossier contenant score.py
    scoring_script="score.py"   # script de scoring
)

# -----------------------------
# Déploiement du modèle
# -----------------------------
deployment_name = f"deploy-{int(time.time())}"

deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=endpoint_name,
    model=model_name,
    code_configuration=code_config,
    environment="AzureML-mlflow-py312-inference",  # environnement MLflow
    instance_type="Standard_DS1_v2",
    instance_count=1,
)

print("➡ Déploiement du modèle...")
ml_client.online_deployments.begin_create_or_update(deployment).result()

# -----------------------------
# Définir ce deployment comme défaut
# -----------------------------
ml_client.online_endpoints.begin_update(
    ManagedOnlineEndpoint(
        name=endpoint_name,
        default_deployment=deployment_name
    )
).result()

print(f"🎉 Modèle déployé avec succès sur l'endpoint : {endpoint_name}")
print(f"➡ Deployment actif : {deployment_name}")
