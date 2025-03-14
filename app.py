import streamlit as st
from poc_RAG import (
    rag_fusion_actualites, 
    rag_fusion_fonds, 
    rag_fusion_fiche_societe_to_word,  
    rag_fusion_multiples_transactions_comparables,
    download_file_from_github,
    add_watermark_to_pdf,
    generate_fiche_societe,
    rag_fusion_fiche_societe_to_word_websearch
)

from langchain.document_loaders import WebBaseLoader
import os
import pandas as pd
import io
import zipfile

# ===============================
# Main Configuration
# ===============================
st.set_page_config(
    page_title="AI for M&A Analysis",
    page_icon="💼",
    layout="centered"
)

# ===============================
# User Registration (Email and Job)
# ===============================
user_data_path = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_MnA\\user_data.csv"

if "registered" not in st.session_state:
    st.session_state["registered"] = False

if "ready_to_access" not in st.session_state:
    st.session_state["ready_to_access"] = False

if not st.session_state["registered"]:
    st.title("Welcome to the AI for M&A Analysis 💼")
    st.write("**Please provide your email and job information to access the application.**")
    with st.form("registration_form"):
        email = st.text_input("Enter your email:", key="email_input")
        job = st.text_input("Enter your profession:", key="job_input")
        submitted = st.form_submit_button("Submit")
    if submitted:
        if email and job:
            try:
                if os.path.exists(user_data_path):
                    user_data = pd.read_csv(user_data_path)
                else:
                    user_data = pd.DataFrame(columns=["email", "job"])
                new_row = pd.DataFrame([{"email": email, "job": job}])
                user_data = pd.concat([user_data, new_row], ignore_index=True)
                user_data.to_csv(user_data_path, index=False)
                st.session_state["registered"] = True
                st.success("Thank you! You can now access the application.")
            except Exception as e:
                st.error(f"An error occurred while saving your data: {e}")
        else:
            st.error("Please fill in both fields.")

if st.session_state["registered"] and not st.session_state["ready_to_access"]:
    st.success("Registration successful! Click the button below to access the application.")
    if st.button("Access Application"):
        st.session_state["ready_to_access"] = True

# ===============================
# Main Application Logic
# ===============================
if st.session_state["registered"] and st.session_state["ready_to_access"]:
    st.title("AI for M&A Analysis 💼")
    st.write("**Simplify your market analysis and research with our intelligent assistant.**")
    
    menu = st.sidebar.radio(
        "Navigation",
        options=[
            "Home", 
            "News", 
            "Company profile", 
            "Funds", 
            "Comparable Transactions", 
            "Watermark PDF"
        ],
        key="main_navigation"
    )
    
    if menu == "Home":
        st.header("Welcome to the AI for M&A Analysis 💡")
        st.write("""
            This application helps you generate company profiles, analyze market news, 
            and access detailed M&A data with the power of AI.
            
            Use the left menu to select the data source you want to query.
        """)
    
    elif menu == "News":
        st.header("Query News Database")
        st.write("Ask a question here, and the AI will query the dedicated news database.")
        question = st.text_input("Ask your question (News):", key="news_question")
        if question:
            st.info("Question received!")
            with st.spinner("Generating answer..."):
                try:
                    answer = rag_fusion_actualites(question)
                    st.success("Answer generated successfully!")
                    st.write("AI-Generated Answer:")
                    st.success(answer)
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
    
    elif menu == "Company profile":
        st.subheader("Générer une fiche entreprise à partir de la réponse du LLM")
        # Saisie manuelle du nom de l'entreprise
        company_name = st.text_input("Entrez le nom de l'entreprise :", key="company_name_input")

        # Bouton 1 : Génération standard
        if st.button("Générer la fiche entreprise"):
            if company_name.strip() == "":
                st.error("Veuillez entrer le nom de l'entreprise.")
            else:
                with st.spinner("Génération de la fiche entreprise en cours..."):
                    try:
                        company_question = f"Fournis-moi une fiche détaillée pour l'entreprise {company_name}."
                        
                        # Interroger le LLM via la fonction RAG dédiée (standard)
                        company_data = rag_fusion_fiche_societe_to_word(company_question)
                        
                        local_template_path = "./Data/Template - Fiche société.docx"
                        github_template_path = "Data/Template - Fiche société.docx"
                        if not os.path.exists(local_template_path):
                            download_file_from_github(github_template_path, local_template_path)
                            st.info("Template téléchargé depuis GitHub.")
                        
                        output_dir = os.path.join(os.getcwd(), "Data", "Fiches")
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"{company_name}_fiche_societe.docx")
                        
                        generate_fiche_societe(company_data, local_template_path, output_path)
                        
                        st.success("Fiche entreprise générée avec succès.")
                        
                        with open(output_path, "rb") as f:
                            doc_bytes = f.read()
                        st.download_button(
                            label="Télécharger la fiche entreprise",
                            data=doc_bytes,
                            file_name=f"{company_name}_fiche_societe.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    except Exception as e:
                        st.error(f"Une erreur est survenue lors de la génération de la fiche entreprise : {e}")

        # Bouton 2 : Génération avec recherche Web
        if st.button("Générer la fiche entreprise (Recherche Web)"):
            if company_name.strip() == "":
                st.error("Veuillez entrer le nom de l'entreprise.")
            else:
                with st.spinner("Génération de la fiche entreprise (avec recherche Web) en cours..."):
                    try:
                        company_question = f"Fournis-moi une fiche détaillée pour l'entreprise {company_name}."
                        
                        # Interroger le LLM via la fonction RAG avec recherche web
                        company_data = rag_fusion_fiche_societe_to_word_websearch(company_question)
                        
                        local_template_path = "./Data/Template - Fiche société.docx"
                        github_template_path = "Data/Template - Fiche société.docx"
                        if not os.path.exists(local_template_path):
                            download_file_from_github(github_template_path, local_template_path)
                            st.info("Template téléchargé depuis GitHub.")
                        
                        output_dir = os.path.join(os.getcwd(), "Data", "Fiches")
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"{company_name}_fiche_societe.docx")
                        
                        generate_fiche_societe(company_data, local_template_path, output_path)
                        
                        st.success("Fiche entreprise (Recherche Web) générée avec succès.")
                        
                        with open(output_path, "rb") as f:
                            doc_bytes = f.read()
                        st.download_button(
                            label="Télécharger la fiche entreprise (Recherche Web)",
                            data=doc_bytes,
                            file_name=f"{company_name}_fiche_societe.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    except Exception as e:
                        st.error(f"Une erreur est survenue lors de la génération de la fiche entreprise (Recherche Web) : {e}")

    elif menu == "Funds":
        st.header("Query the Funds Database")
        st.write("Ask a question here, and the AI will query the dedicated funds database.")
        question = st.text_input("Ask your question (Funds):", key="funds_question")
        if question:
            st.info("Question received!")
            with st.spinner("Generating answer..."):
                try:
                    answer = rag_fusion_fonds(question)
                    st.success("Answer generated successfully!")
                    st.write("AI-Generated Answer:")
                    st.success(answer)
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
    
    elif menu == "Comparable Transactions":
        st.header("Query Comparable Transactions Database")
        st.write("Ask a question here, and the AI will query the database dedicated to comparable transactions and financial multiples.")
        question = st.text_input("Ask your question (Comparable Transactions):", key="transactions_question")
        if question:
            st.info("Question received!")
            with st.spinner("Generating answer..."):
                try:
                    answer = rag_fusion_multiples_transactions_comparables(question)
                    st.success("Answer generated successfully!")
                    st.write("AI-Generated Answer:")
                    st.success(answer)
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
    
    elif menu == "Watermark PDF":
        st.header("Add Watermark to PDF")
        st.write("Upload a PDF and add bank names. For each bank name, the watermark will be formatted as 'Confidentiel - Bank Name'.")
        uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], key="uploaded_pdf")
        if uploaded_pdf is not None:
            pdf_bytes = uploaded_pdf.read()
            st.write(f"Uploaded file: {uploaded_pdf.name} ({len(pdf_bytes)} bytes)")
            st.session_state.original_pdf_name = uploaded_pdf.name
            bank_name = st.text_input("Enter a bank name (e.g., CIC, BNP):", key="bank_name_input_pdf")
            if st.button("Add Bank Name"):
                if bank_name:
                    if "bank_names" not in st.session_state:
                        st.session_state.bank_names = []
                    st.session_state.bank_names.append(bank_name.strip())
                    st.success(f"Bank name '{bank_name.strip()}' added.")
                else:
                    st.error("Please enter a bank name.")
            if "bank_names" in st.session_state and st.session_state.bank_names:
                st.markdown("**Bank names added:**")
                for name in st.session_state.bank_names:
                    st.markdown(f"- **{name}**")
            if st.button("Generate Watermarked PDFs"):
                if "bank_names" in st.session_state and st.session_state.bank_names:
                    with st.spinner("Generating watermarked PDFs..."):
                        try:
                            st.session_state.watermarked_pdfs = []
                            for bank in st.session_state.bank_names:
                                watermarked_pdf_bytes = add_watermark_to_pdf(pdf_bytes, bank)
                                base_name = os.path.splitext(st.session_state.original_pdf_name)[0]
                                output_filename = f"{base_name}_{bank}.pdf"
                                st.session_state.watermarked_pdfs.append((bank, watermarked_pdf_bytes, output_filename))
                            st.success("Watermarked PDFs generated successfully!")
                        except Exception as e:
                            st.error(f"An error occurred while generating watermarked PDFs: {e}")
                else:
                    st.error("Please add at least one bank name before generating PDFs.")
            if "watermarked_pdfs" in st.session_state and st.session_state.watermarked_pdfs:
                with st.spinner("Creating ZIP archive..."):
                    try:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for _, pdf_data, output_filename in st.session_state.watermarked_pdfs:
                                zip_file.writestr(output_filename, pdf_data)
                        zip_buffer.seek(0)
                        st.download_button(
                            label="Download All Watermarked PDFs",
                            data=zip_buffer,
                            file_name="watermarked_pdfs.zip",
                            mime="application/zip"
                        )
                    except Exception as e:
                        st.error(f"An error occurred while creating the ZIP archive: {e}")