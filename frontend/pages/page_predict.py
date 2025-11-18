import streamlit as st
import requests

# Définir l'URL du backend
API_URL= API_URL = "http://localhost:8000/predict"

st.title("Bienvenue sur la page Predict")

#Définir les champs de saisies
st.subheader("Veuillez entrez les caractèristiques")

# En paramètre du slider, dans l'ordre : min_value, max_value, default_value, step
sepal_length= st.number_input("Longueur du sépale (cm)", 4.0, 8.0, 4.0)
sepal_width = st.number_input("Largeur du sépale (cm)", 2.0, 5.0, 2.0)
petal_length=st.number_input("Longueur de pétale (cm)", 1.0, 7.0, 1.0)
petal_width=st.number_input("Largeur de pétale (cm)", 0.1, 3.0, 0.1)

#Appel API + cache si même valeurs
@st.cache_data
def call_api(payload):
    response=requests.post(API_URL, json=payload)
    if response.status_code ==200:
        return response.json()
    else:
        return {"error": response.text}
    
if st.button("Prédire"): # À chaque clic, Streamlit prépare le payload
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    # appel API
    result= call_api(payload)

    # affiche le résultat s'il n'y'a pas d'erreur
    if "error" in result:
        st.error(f"Erreur API : {result['error']}")
    else:
        st.success(f"Variété prédite : **{result['variety']}** (code : {result['prediction']})")