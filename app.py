import streamlit as st
from poc_RAG import (
    rag_fusion, rag_fusion_actualites, rag_fusion_fonds,
    rag_fusion_multiples_transactions_comparables, add_watermark_to_pdf
)
from langchain.document_loaders import WebBaseLoader
import os
import pandas as pd

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
        options=["Home", "Comprehensive Data (News+Companies+Funds)", "News", "Funds", "Comparable Transactions", "Watermark PDF"],
        key="main_navigation"
    )
    
    if menu == "Home":
        st.header("Welcome to the AI for M&A Analysis 💡")
        st.write("""
            This application helps you generate company profiles, analyze market news, 
            and access detailed M&A data with the power of AI.
            
            Use the left menu to select the data source you want to query.
        """)
    
    elif menu == "Comprehensive Data (News+Companies+Funds)":
        st.header("Query Comprehensive Data (News, Companies, Funds)")
        st.write("Ask a question here, and the AI will query the full database, including news, companies, and funds.")
        question = st.text_input("Ask your question:", key="comprehensive_data")
        if question:
            st.info("Question received!")
            with st.spinner("Generating answer..."):
                try:
                    answer = rag_fusion(question)
                    st.success("Answer generated successfully!")
                    st.write("AI-Generated Answer:")
                    st.success(answer)
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
    
    elif menu == "News":
        st.header("Query News Database or Generate a Company Profile")
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
        st.write("Upload a PDF and add one or more watermark texts to generate watermarked versions for download.")
        uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], key="uploaded_pdf")
        if uploaded_pdf is not None:
            pdf_bytes = uploaded_pdf.read()
            st.write(f"Uploaded file: {uploaded_pdf.name} ({len(pdf_bytes)} bytes)")
            watermark_text = st.text_input("Enter watermark text (e.g., 'Confidentiel - CIC')", key="watermark_text")
            if st.button("Add Watermark"):
                if watermark_text:
                    with st.spinner("Generating watermarked PDF..."):
                        try:
                            watermarked_pdf_bytes = add_watermark_to_pdf(pdf_bytes, watermark_text)
                            if "watermarked_pdfs" not in st.session_state:
                                st.session_state["watermarked_pdfs"] = []
                            st.session_state["watermarked_pdfs"].append((watermark_text, watermarked_pdf_bytes))
                            st.success(f"Watermarked PDF generated with watermark: {watermark_text}")
                        except Exception as e:
                            st.error(f"An error occurred while generating the watermarked PDF: {e}")
                else:
                    st.error("Please enter a watermark text.")
            if "watermarked_pdfs" in st.session_state and st.session_state["watermarked_pdfs"]:
                st.subheader("Generated Watermarked PDFs")
                for i, (text, pdf_data) in enumerate(st.session_state["watermarked_pdfs"]):
                    st.write(f"Watermark: {text}")
                    st.download_button(label="Download Watermarked PDF", data=pdf_data, file_name=f"watermarked_{i+1}.pdf", mime="application/pdf")
