
       import os
import streamlit as st

st.title("Classification des fleurs")
import tensorflow as tf

from PIL import Image

# Charger le modèle
model = tf.keras.models.load_model('cnn_model_5_classes.h5')

# Interface utilisateur
st.title("Classificateur d'images")

# Recompiler le modèle avec les métriques souhaitées
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


uploaded_file = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Charger l'image et la prétraiter
    image = Image.open(uploaded_file)
