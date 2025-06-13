import streamlit as st
import os
import io
import zipfile
import pandas as pd
from poc_RAG import (
    rag_fusion_actualites, 
    rag_fusion_fonds, 
    rag_fusion_fiche_societe_to_word,  
    rag_fusion_multiples_transactions_comparables,
    download_file_from_github,
    add_watermark_to_pdf,
    generate_fiche_societe,
    rag_fusion_fiche_societe_to_word_websearch,
    password_break,
    rag_fusion_actualites_search_preview
)
from langchain.document_loaders import WebBaseLoader

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
user_data_path = "C:\\Users\\namar.DA-CF\\OneDrive - D&A Corporate Finance\\Documents\\poc_RAG\\Projet_test\\RAG_MnA\\user_data.csv"

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

            # Bouton pour la base interne
            if st.button("Generate Answer (Base DB)", key="news_generate_base"):
                with st.spinner("Generating answer from internal news DB..."):
                    try:
                        answer = rag_fusion_actualites(question)
                        st.success("Answer generated successfully!")
                        st.write("AI-Generated Answer:")
                        st.success(answer)
                    except Exception as e:
                        st.error(f"An error occurred while generating the answer: {e}")

            # Bouton pour la web search
            if st.button("Generate Answer (Web Search)", key="news_generate_web"):
                with st.spinner("Generating answer via Web Search Preview..."):
                    try:
                        answer = rag_fusion_actualites_search_preview(question)
                        st.success("Web-search answer generated successfully!")
                        st.write("AI-Generated Answer (Web Search):")
                        st.success(answer)
                    except Exception as e:
                        st.error(f"An error occurred while generating the web-search answer: {e}")

    elif menu == "Company profile":
        st.subheader("Generate a company profile")
        st.write("Don't hesitate to ask for both profiles (with and without Web Search) for a more complete answer!")
        company_name = st.text_input("Enter the company name:", key="company_name_input")

        # Button for standard generation
        if st.button("Generate Company Profile"):
            if company_name.strip() == "":
                st.error("Please enter the company name.")
            else:
                with st.spinner("Generating company profile..."):
                    try:
                        company_question = f"Fournis-moi une fiche détaillée pour l'entreprise {company_name}."
                        company_data = rag_fusion_fiche_societe_to_word(company_question)
                        local_template_path = "./Data/Template - Fiche société.docx"
                        github_template_path = "Data/Template - Fiche société.docx"
                        if not os.path.exists(local_template_path):
                            download_file_from_github(github_template_path, local_template_path)
                            st.info("Template downloaded from GitHub.")
                        output_dir = os.path.join(os.getcwd(), "Data", "Fiches")
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"{company_name}_fiche_societe.docx")
                        generate_fiche_societe(company_data, local_template_path, output_path)
                        st.success("Company profile generated successfully!")
                        with open(output_path, "rb") as f:
                            doc_bytes = f.read()
                        st.download_button(
                            label="Download Company Profile",
                            data=doc_bytes,
                            file_name=f"{company_name}_fiche_societe.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        # Button for generation with Web Search
        if st.button("Generate Company Profile (Web Search)"):
            if company_name.strip() == "":
                st.error("Please enter the company name.")
            else:
                with st.spinner("Generating company profile (with Web Search)..."):
                    try:
                        company_question = f"Fournis-moi une fiche détaillée pour l'entreprise {company_name}."
                        company_data = rag_fusion_fiche_societe_to_word_websearch(company_question)
                        local_template_path = "./Data/Template - Fiche société.docx"
                        github_template_path = "Data/Template - Fiche société.docx"
                        if not os.path.exists(local_template_path):
                            download_file_from_github(github_template_path, local_template_path)
                            st.info("Template downloaded from GitHub.")
                        output_dir = os.path.join(os.getcwd(), "Data", "Fiches")
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"{company_name}_fiche_societe.docx")
                        generate_fiche_societe(company_data, local_template_path, output_path)
                        st.success("Company profile (Web Search) generated successfully!")
                        with open(output_path, "rb") as f:
                            doc_bytes = f.read()
                        st.download_button(
                            label="Download Company Profile (Web Search)",
                            data=doc_bytes,
                            file_name=f"{company_name}_fiche_societe.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

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
        st.write("Upload one or more PDFs and add bank names. For each bank name, the watermark will be formatted as 'Confidentiel - Bank Name'.")
        
        # Permettre l'upload multiple de PDF
        uploaded_pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True, key="uploaded_pdfs")
        if uploaded_pdfs:
            # Stocker les PDFs uploadés dans la session
            st.session_state.original_pdfs = []
            for pdf in uploaded_pdfs:
                pdf_bytes = pdf.read()
                st.write(f"Uploaded file: {pdf.name} ({len(pdf_bytes)} bytes)")
                st.session_state.original_pdfs.append((pdf.name, pdf_bytes))
        
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
            if "bank_names" in st.session_state and st.session_state.bank_names and "original_pdfs" in st.session_state and st.session_state.original_pdfs:
                with st.spinner("Generating watermarked PDFs..."):
                    try:
                        st.session_state.watermarked_pdfs = []  # Liste des tuples: (original_filename, bank, watermarked_pdf_bytes, output_filename)
                        for original_name, pdf_bytes in st.session_state.original_pdfs:
                            base_name = os.path.splitext(original_name)[0]
                            for bank in st.session_state.bank_names:
                                watermarked_pdf_bytes = add_watermark_to_pdf(pdf_bytes, bank)
                                output_filename = f"{base_name}_{bank}.pdf"
                                st.session_state.watermarked_pdfs.append((original_name, bank, watermarked_pdf_bytes, output_filename))
                                st.success(f"Watermark added for '{original_name}' with bank '{bank}'.")
                    except Exception as e:
                        st.error(f"An error occurred while generating watermarked PDFs: {e}")
            else:
                st.error("Please upload at least one PDF and add at least one bank name before generating PDFs.")
        
        if st.button("Break Adobe Password"):
            if "original_pdfs" in st.session_state and st.session_state.original_pdfs:
                with st.spinner("Breaking passwords..."):
                    try:
                        st.session_state.watermarked_pdfs = []  # tuples (fichier, pdf_bytes, nom_de_sortie)
                        for orig_name, pdf_bytes in st.session_state.original_pdfs:
                            base = os.path.splitext(orig_name)[0]
                            wm_bytes = password_break(pdf_bytes)
                            out_name = f"{base}.pdf"
                            st.session_state.watermarked_pdfs.append((orig_name, wm_bytes, out_name))
                            st.success(f"Passwrod broken for « {orig_name} »")
                    except Exception as e:
                        st.error(f"An error happened: {e}")
            else:
                st.error("Please upload at least one PDF before breaking passwords.")

        if "watermarked_pdfs" in st.session_state and st.session_state.watermarked_pdfs:
            with st.spinner("Creating ZIP archive..."):
                try:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        # Désassemblage en 3 éléments : (original_name, pdf_bytes, output_filename)
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
