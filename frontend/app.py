import streamlit as st
import pandas as pd

st.set_page_config(page_title="Iris Predict")
st.title("Iris Predict - Interface web")
st.write("""
Bienvenue dans **Iris Predict** !

Cette application permet de prédire la variété d'une fleur d'Iris grâce :
- à un modèle ML entraîné,
- à une API backend construite avec FastAPI,
- à une interface utilisateur Streamlit.

Rendez-vous dans la page **Prédiction** pour tester le modèle.
""")

st.write("Utilisez le menu à gauche pour accéder à la page de prédiction.")
