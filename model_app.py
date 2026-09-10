"""Formulaire Streamlit servant le même modèle que le service web.

Lancement : streamlit run model_app.py
"""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "regression.joblib"
COLUMNS = ["size", "nb_rooms", "garden"]


@st.cache_resource
def load_model():
    """Charge le modèle une seule fois pour toute la session Streamlit."""
    return joblib.load(MODEL_PATH)


st.title("Estimation du prix d'une maison")
st.write(
    "Renseignez les caractéristiques du bien. L'estimation provient du même "
    "modèle de régression linéaire que celui servi par l'API."
)

size = st.number_input(
    "Surface habitable (m2)", min_value=10.0, max_value=500.0, value=120.0, step=5.0
)
nb_rooms = st.number_input("Nombre de chambres", min_value=0, max_value=10, value=3, step=1)
garden = st.number_input("Jardin (0 = non, 1 = oui)", min_value=0, max_value=1, value=0, step=1)

if st.button("Estimer le prix"):
    model = load_model()
    features = pd.DataFrame([[size, nb_rooms, garden]], columns=COLUMNS)
    price = float(model.predict(features)[0])
    st.write(f"Prix estimé : {price:,.2f}".replace(",", " "))
