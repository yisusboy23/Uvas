
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="Sistema Vitivinícola")

st.title("🍇 Sistema Inteligente para Uvas en Tarija")
st.markdown("Predicción del destino, madurez y calendario óptimo de cosecha.")

brix = st.slider("Brix (azúcar)", 18.0, 26.0, 22.0)
ph = st.slider("pH", 3.0, 4.0, 3.5)
acidez = st.slider("Acidez total", 4.5, 7.5, 6.0)
temp = st.slider("Temperatura", 15, 35, 25)
humedad = st.slider("Humedad", 40, 90, 60)
lluvia = st.slider("Lluvia", 0, 60, 10)
suelo = st.selectbox("Suelo", ["Arcilloso", "Calcáreo"])
variedad = st.selectbox("Variedad", ["Moscatel", "Tannat", "Red Globe"])

suelo_val = 0 if suelo == "Arcilloso" else 1
var_map = {"Moscatel": 0, "Tannat": 1, "Red Globe": 2}

entrada = np.array([[brix, ph, acidez, temp, humedad, lluvia, suelo_val, var_map[variedad]]])

modelo = tf.keras.models.load_model("modelo.h5")
arbol = joblib.load("arbol.pkl")
escalador = joblib.load("escalador.pkl")
entrada_scaled = escalador.transform(entrada)

destinos = {0: "🍷 Vino Tinto", 1: "🥂 Vino Blanco", 2: "🍸 Singani", 3: "🍇 Uva de Mesa"}

pred_rna = np.argmax(modelo.predict(entrada_scaled), axis=1)[0]
pred_tree = arbol.predict(entrada_scaled)[0]

st.subheader("📊 Resultados")
st.write("🔺 Red Neuronal:", destinos[pred_rna])
st.write("🔸 Árbol de Decisión:", destinos[pred_tree])
st.write("🔹 Madurez estimada (demo): Alta")
st.write("📅 Calendario sugerido (demo): [28, 30, 32, 35, 33]")
