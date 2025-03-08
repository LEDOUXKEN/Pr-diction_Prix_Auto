# Importation des bibliothèques nécessaires

# os : Permet d'interagir avec le système d'exploitation. Utilisé pour manipuler les chemins de fichiers.
import os

# pandas : Bibliothèque pour la manipulation et l'analyse de données. Elle permet de travailler facilement avec des DataFrames.
import pandas as pd

# streamlit : Framework permettant de créer des applications web interactives. Il est utilisé ici pour l'interface utilisateur.
import streamlit as st

# numpy : Bibliothèque pour la manipulation de tableaux multidimensionnels et de calculs numériques. Elle est utilisée pour préparer les données avant la prédiction.
import numpy as np

# joblib : Permet de sérialiser et de désérialiser des objets Python. Ici, on l'utilise pour charger le modèle pré-entraîné de Machine Learning.
import joblib

# altair : Bibliothèque de visualisation de données déclarative. Elle permet de créer des graphiques interactifs pour mieux comprendre les données.
import altair as alt

# --- TITRE ET DESCRIPTION ---
# Affichage du titre et de la description de l'application avec un style HTML personnalisé.
st.markdown("<h1 style='color:blue; font-weight:bold;'>Prédiction du Prix d'une Voiture</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color:green;'>Application réalisée par <b>Fidèle Ledoux</b></h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size:18px;'>Cette application utilise un modèle de Machine Learning pour prédire le prix d'une voiture en fonction de ses caractéristiques.</p>", unsafe_allow_html=True)

# --- CHARGEMENT DU MODÈLE ---
# Charger le modèle de Machine Learning pré-entraîné. Ce modèle a été sauvegardé sous le nom "final_model.joblib".
# joblib est utilisé ici pour charger le modèle sérialisé depuis un fichier.
model = joblib.load("final_model.joblib")

# --- FONCTION DE PRÉDICTION ---
# Cette fonction prend en entrée les caractéristiques de la voiture et prédit son prix en utilisant le modèle de Machine Learning.
def inference(wheel_base, length, width, curb_weight, engine_size, horsepower, city_mpg, highway_mpg, peak_rpm):
    # Créer un tableau numpy avec les caractéristiques saisies par l'utilisateur
    new_data = np.array([wheel_base, length, width, curb_weight, engine_size, 
                         horsepower, city_mpg, highway_mpg, peak_rpm])
    # Appliquer le modèle pour prédire le prix de la voiture
    pred = model.predict(new_data.reshape(1, -1))
    return pred[0]

# --- SIDEBAR : INPUTS UTILISATEUR ---
# Affichage des champs de saisie interactifs pour permettre à l'utilisateur d'entrer les caractéristiques de la voiture.
st.sidebar.header("Entrez les Caractéristiques de la Voiture")

# Inputs numériques : Ces champs permettent à l'utilisateur de saisir des valeurs numériques pour chaque caractéristique.
wheel_base = st.sidebar.number_input("Empattement (wheel_base) :", min_value=50, max_value=130, value=90)
length = st.sidebar.number_input('Longueur (length) :', min_value=100, max_value=200, value=150)
width = st.sidebar.number_input('Largeur (width) :', min_value=50, max_value=100, value=65)
curb_weight = st.sidebar.number_input('Poids à vide (curb_weight) :', min_value=500, max_value=5000, value=2000)
engine_size = st.sidebar.number_input('Taille du moteur (engine_size) :', min_value=50, max_value=500, value=120)
horsepower = st.sidebar.number_input('Puissance du moteur (horsepower) :', min_value=50, max_value=1000, value=110)
city_mpg = st.sidebar.number_input('Consommation en ville (city_mpg) :', min_value=10, max_value=100, value=20)
highway_mpg = st.sidebar.number_input('Consommation sur autoroute (highway_mpg) :', min_value=10, max_value=100, value=30)
peak_rpm = st.sidebar.number_input('Régime moteur maximal (peak_rpm) :', min_value=1000, max_value=10000, value=5000)

# --- RÉSUMÉ DES DONNÉES SAISIES ---
# Après que l'utilisateur a saisi les informations, elles sont affichées sous forme de tableau pour qu'il puisse les vérifier.
input_data = {
    "Empattement": wheel_base,
    "Longueur": length,
    "Largeur": width,
    "Poids à vide": curb_weight,
    "Taille du moteur": engine_size,
    "Puissance du moteur": horsepower,
    "Consommation en ville": city_mpg,
    "Consommation sur autoroute": highway_mpg,
    "Régime moteur maximal": peak_rpm
}
# Utilisation de pandas pour afficher les informations sous forme de DataFrame (tableau).
st.write(pd.DataFrame(input_data, index=[0]))

# --- PREDICTION ---
# Lorsque l'utilisateur clique sur le bouton "Prédire le Prix de la Voiture", la prédiction est effectuée.
# Le résultat est ensuite affiché en couleur pour rendre l'information plus lisible.
if st.button("Prédire le Prix de la Voiture"):
    try:
        # Appel de la fonction inference pour obtenir le prix estimé
        prediction = inference(wheel_base, length, width, curb_weight, engine_size, 
                               horsepower, city_mpg, highway_mpg, peak_rpm)
        # Affichage de la prédiction avec un message de succès et un formatage coloré
        st.subheader("Prédiction du Prix :")
        st.markdown(f"<h3 style='color:green;'>Le prix estimé de la voiture est : <b>{prediction:.2f} dollars</b></h3>", unsafe_allow_html=True)
    except Exception as e:
        # En cas d'erreur, afficher un message d'erreur
        st.error(f"Erreur lors de la prédiction : {str(e)}")

# --- VISUALISATION INTERACTIVE ---
# Cette section utilise la bibliothèque altair pour créer un graphique en barres montrant les caractéristiques de la voiture saisies par l'utilisateur.
# altair est une bibliothèque de visualisation déclarative qui permet de créer des graphiques interactifs.
st.subheader("Visualisation Interactive des Caractéristiques")
# Création d'un DataFrame sous forme longue pour une visualisation avec altair
df_melt = pd.DataFrame(input_data, index=[0]).melt(var_name="Caractéristique", value_name="Valeur")
# Utilisation de altair pour générer un graphique en barres avec les caractéristiques de la voiture.
chart = alt.Chart(df_melt).mark_bar().encode(
    x=alt.X("Caractéristique:N", sort=None, title="Caractéristiques"),
    y=alt.Y("Valeur:Q", title="Valeur"),
    color=alt.Color("Caractéristique:N", legend=None),
    tooltip=["Caractéristique", "Valeur"]
).properties(
    width=800,
    height=400
)
# Affichage du graphique interactif
st.altair_chart(chart, use_container_width=True)

# --- MESSAGE DE RETOUR ET DESIGN AMÉLIORÉ ---
# Ajout d'un petit texte explicatif concernant la prédiction du prix de la voiture.
st.markdown("<p style='font-size:16px;'>N'oubliez pas que cette prédiction repose sur un modèle de Machine Learning entraîné avec des données historiques. Les résultats sont des estimations et peuvent varier en fonction de plusieurs facteurs.</p>", unsafe_allow_html=True)
