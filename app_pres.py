import streamlit as st
from poc_RAG import rag_fusion, rag_fusion_actualites, rag_fusion_fonds, rag_fusion_fiche_societe_to_word, generate_fiche_societe, rag_fusion_multiples_transactions_comparables  # Importer votre fonction rag_fusion
from langchain.document_loaders import WebBaseLoader  
import os  

#from poc_RAG import rag_fusion_all, rag_fusion_actualites, rag_fusion_fonds  
# Assurez-vous de créer ou d'importer vos fonctions qui interrogent les données segmentées.
# Exemple :
# rag_fusion_all() -> interroge la base Actualités + Entreprises + Fonds
# rag_fusion_actualites() -> interroge la base dédiée aux actualités
# rag_fusion_fonds() -> interroge la base dédiée aux fonds

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
    options=["Accueil", "Données complètes (Actu+Entreprises+Fonds)", "Actualités", "Fonds", "Transactions comparables", "Analyse avancée"]
)

# ===============================
# Logique pour chaque menu
# ===============================
if menu == "Accueil":
    st.header("Bienvenue sur l'application IA pour le M&A 💡")
    st.write(
        """
        Cette application vous aide à générer des fiches sur des entreprises, des fonds 
        ou à analyser les actualités liées au M&A grâce à l'IA.
        
        Utilisez le menu de gauche pour sélectionner la source de données que vous souhaitez interroger.
        """
    )

elif menu == "Données complètes (Actu+Entreprises+Fonds)":
    st.header("Interroger la base complète (Actualités, Entreprises, Fonds)")
    st.write(
        """
        Posez une question ici, l'IA interrogera la base complète qui comprend 
        les actualités, les entreprises et les fonds.
        """
    )
    
    question = st.text_input("Posez votre question :", "")
    if question:
        try:
            # Appeler la fonction rag_fusion_all avec la question
            answer = rag_fusion(question)
            st.write("Réponse générée par l'IA :")
            st.success(answer)
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la génération de la réponse : {e}")

elif menu == "Actualités":
    st.header("Interroger la base d'actualités ou demander une fiche")
    st.write(
        """
        Posez une question ici, l'IA interrogera uniquement la base dédiée aux actualités.
        """
    )
    
    question = st.text_input("Posez votre question ou demandez votre fiche (Actualités) :", "")
    if question:
        try:
            # Appeler la fonction rag_fusion_actualites
            answer = rag_fusion_actualites(question)
            st.write("Réponse générée par l'IA :")
            st.success(answer)
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la génération de la réponse : {e}")
    
    # --- Ajouter la fonctionnalité de génération de fiche entreprise ---
    st.subheader("Générer une fiche entreprise à partir de la réponse du LLM")
    
    # Saisie manuelle du nom de l'entreprise
    company_name = st.text_input("Entrez le nom de l'entreprise :", "")
    
    if st.button("Générer la fiche entreprise"):
        if company_name.strip() == "":
            st.error("Veuillez entrer le nom de l'entreprise.")
        else:
            try:
                # Formulez une question pertinente pour obtenir les informations de l'entreprise
                company_question = f"Fournis-moi une fiche détaillée pour l'entreprise {company_name}."
                
                # Interroger le LLM via la fonction RAG modifiée
                company_data = rag_fusion_fiche_societe_to_word(company_question)
                
                # Chemin vers le template Word
                template_path = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_MnA\\Data\\Template - Fiche société.docx"
                    
                # Vérifiez si le template existe
                if not os.path.exists(template_path):
                    st.error(f"Le fichier template Word n'a pas été trouvé à : {template_path}")
                else:
                    # Chemin de sortie
                    output_dir = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_MnA\\Data\\Fiches"
                    os.makedirs(output_dir, exist_ok=True)  # Crée le dossier s'il n'existe pas
                    output_path = os.path.join(output_dir, f"{company_name}_fiche_societe.docx")
                    
                    # Remplir le template avec les données du LLM
                    generate_fiche_societe(company_data, template_path, output_path)
                    
                    # Lire le fichier en bytes pour le télécharger
                    with open(output_path, "rb") as f:
                        doc_bytes = f.read()
                    st.download_button(
                        label="Télécharger la fiche entreprise",
                        data=doc_bytes,
                        file_name=f"{company_name}_fiche_societe.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                    st.success("Fiche entreprise générée avec succès.")
            except Exception as e:
                st.error(f"Une erreur est survenue lors de la génération de la fiche entreprise : {e}")

elif menu == "Fonds":
    st.header("Interroger la base de données sur les fonds")
    st.write(
        """
        Posez une question ici, l'IA interrogera uniquement la base dédiée aux fonds.
        """
    )
    
    question = st.text_input("Posez votre question (Fonds) :", "")
    if question:
        try:
            # Appeler la fonction rag_fusion_fonds
            answer = rag_fusion_fonds(question)
            st.write("Réponse générée par l'IA :")
            st.success(answer)
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la génération de la réponse : {e}")

elif menu == "Transactions comparables":
    st.header("Interroger la base de données sur les transactions comparables et les multiples")
    st.write(
        """
        Posez une question ici, l'IA interrogera uniquement la base dédiée aux transactions comparables 
        et aux multiples financiers (e.g., EV/Revenue, EV/EBITDA, EV/EBIT). 
        Obtenez des informations pertinentes sur des transactions similaires, leurs caractéristiques et leurs multiples.
        """
    )
    
    question = st.text_input("Posez votre question (Transactions comparables) :", "")
    if question:
        try:
            # Appeler la fonction rag_fusion_multiples_transactions_comparables
            answer = rag_fusion_multiples_transactions_comparables(question)
            st.write("Réponse générée par l'IA :")
            st.success(answer)
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la génération de la réponse : {e}")


elif menu == "Analyse avancée":
    st.header("Analyse avancée 🔍")
    st.write(
        """
        Cette section sera dédiée à des analyses complexes et des visualisations avancées. 
        Fonctionnalités à venir dans une prochaine version !
        """
    )