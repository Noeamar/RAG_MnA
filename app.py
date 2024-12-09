import streamlit as st
from poc_RAG import rag_fusion  # Importer votre fonction rag_fusion
from langchain.document_loaders import WebBaseLoader    

# ===============================
# Configuration principale
# ===============================
st.set_page_config(
    page_title="IA pour le M&A",
    page_icon="💼",
    layout="centered"
)

# ===============================
# En-tête de l'application
# ===============================
st.title("L'IA pour le M&A 💼")
st.write("**Simplifiez votre analyse de marché et vos recherches grâce à notre assistant intelligent.**")

# ===============================
# Menu de navigation
# ===============================
menu = st.sidebar.radio(
    "Navigation",
    options=["Accueil", "RAG Fusion", "Analyse avancée"]
)

# ===============================
# Logique pour chaque menu
# ===============================
if menu == "Accueil":
    # Page d'accueil
    st.header("Bienvenue sur l'application IA pour le M&A 💡")
    st.write(
        """
        Cette application vous aide à générer des fiches sur des entreprises et des marchés 
        grâce à l'intelligence artificielle. Naviguez à travers les menus pour découvrir ses fonctionnalités.
        """
    )

elif menu == "RAG Fusion":
    # Page RAG Fusion
    st.header("Générer des fiches société ou des fiches de marché 📄")
    st.write(
        """
        **Utilisez cette section pour poser vos questions et obtenir des informations 
        détaillées sur des entreprises ou des marchés spécifiques.** 
        """
    )
    
    # Entrée de question utilisateur
    question = st.text_input("Posez votre question :", "")
    if question:
        try:
            # Appeler la fonction rag_fusion avec la question
            answer = rag_fusion(question)
            st.write("Réponse générée par l'IA :")
            st.success(answer)
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la génération de la réponse : {e}")

elif menu == "Analyse avancée":
    # Page Analyse avancée
    st.header("Analyse avancée 🔍")
    st.write(
        """
        Cette section sera dédiée à des analyses complexes et des visualisations avancées. 
        Fonctionnalités à venir dans une prochaine version !
        """
    )
