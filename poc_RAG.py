#!/usr/bin/env python
# coding: utf-8

import os
import json
from json import dumps, loads
from operator import itemgetter
import datetime
import pandas as pd
import bs4

# Importations de LangChain et autres
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from docxtpl import DocxTemplate

# Pour télécharger depuis Google Cloud Storage
from google.cloud import storage

# --- Configuration des variables d'environnement ---
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'
# On récupère la clé OPENAI_API_KEY depuis l'environnement (définie via les secrets sur Streamlit Cloud par exemple)
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
    raise ValueError("La variable d'environnement OPENAI_API_KEY n'est pas définie. Veuillez la définir dans vos secrets.")
os.environ['OPENAI_API_KEY'] = openai_api_key

# Nom du bucket GCS (vérifiez que vos credentials sont configurés)
GCS_BUCKET = "rag-mna_cloudbuild"


def download_file_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """
    Télécharge un fichier depuis un bucket Google Cloud Storage vers un chemin local.
    """
    print(f"[LOG] Initialisation du client GCS pour télécharger {source_blob_name}...", flush=True)
    client = storage.Client()  # Utilise les credentials configurés via GOOGLE_APPLICATION_CREDENTIALS
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    print(f"[LOG] Téléchargement de gs://{bucket_name}/{source_blob_name} vers {destination_file_name}...", flush=True)
    blob.download_to_filename(destination_file_name)
    
    # Définir les permissions pour que le fichier soit lisible
    os.chmod(destination_file_name, 0o644)
    print("[LOG] Téléchargement terminé et permissions définies.", flush=True)


# --- Fonction RAG Fusion (Comprehensive Data: News+Companies+Funds) ---
def rag_fusion(question: str) -> str:
    print("[LOG] Démarrage de rag_fusion pour la question :", question)
    # On définit le répertoire local qui contient l'index.
    local_index_dir = "./Data/FAISS_index"
    local_index_file = os.path.join(local_index_dir, "index.faiss")
    gcs_blob_path = "FAISS_index/index.faiss"
    if not os.path.exists(local_index_file):
        download_file_from_gcs(GCS_BUCKET, gcs_blob_path, local_index_file)
    else:
        print(f"[LOG] Fichier index déjà présent localement : {local_index_file}")
    
    embedding = OpenAIEmbeddings()
    # FAISS.load_local attend le répertoire contenant le fichier "index.faiss"
    vectorstore = FAISS.load_local(local_index_dir, embeddings=embedding, allow_dangerous_deserialization=True)
    print("[LOG] Index FAISS chargé.")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "score_threshold": 0.01})
    
    query_generation_template = """You are a seasoned M&A consultant with access to a broad dataset that includes recent news, company profiles, and investment fund details. Given the user's question:

{question}

Generate exactly 4 focused queries that will help retrieve the most relevant and comprehensive information from this rich dataset. The queries should cover aspects of M&A news, company insights, and fund activities if relevant. Aim to find highly relevant and up-to-date details that answer the user's inquiry.

Output 4 queries:
"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)
    generate_queries = (prompt_rag_fusion
                        | ChatOpenAI(model='o1-mini')
                        | StrOutputParser()
                        | (lambda x: x.split("\n")))
    queries = generate_queries.invoke({"question": question})
    print("[LOG] Requêtes générées :", queries)
    
    results = [retriever.invoke(q) for q in queries]
    print("[LOG] Documents récupérés :", results)
    
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            doc_str = dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)
    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [loads(d_str)]
    ]
    print(f"[LOG] Documents fusionnés : {len(reranked_docs)} documents rerankés.")
    
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])
    
    answer_template = """You are an M&A expert who can analyze recent news, company data, and fund information to provide a comprehensive and accurate answer. Using the following context extracted from various M&A-related sources (news, companies, funds), answer the user's question concisely and factually. Highlight relevant deals, company details, or fund strategies if mentioned. Do not invent information that isn't provided. Always give your source. It's in the name of the document you are using the information (it's Arx or CFNews. Do not say that it comes from the context, always say that it is from Arx or CFNews from the name of the doc in the metadata).

Context:
{context}

Question: {question}

Provide a clear, fact-based answer focusing on the M&A domain.
"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')
    final_input = {"context": context, "question": question}
    answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
    
    print("[LOG] Réponse générée.")
    return answer


# --- Fonction RAG Fusion Actualités ---
def rag_fusion_actualites(question: str) -> str:
    print("[LOG] Démarrage de rag_fusion_actualites pour la question :", question)
    local_index_dir = "./Data/FAISS_index_actualites"
    local_index_file = os.path.join(local_index_dir, "index.faiss")
    gcs_blob_path = "FAISS_index_actualites/index.faiss"
    if not os.path.exists(local_index_file):
        download_file_from_gcs(GCS_BUCKET, gcs_blob_path, local_index_file)
    else:
        print(f"[LOG] Fichier index actualités déjà présent : {local_index_file}")
    
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(local_index_dir, embeddings=embedding, allow_dangerous_deserialization=True)
    print("[LOG] Index actualités chargé.")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "score_threshold": 0.01})
    
    query_generation_template = """You are a knowledgeable M&A news analyst. Your role is to generate multiple targeted search queries to retrieve the most relevant and recent M&A news from a specialized news database.

Given the user's question: {question}

Generate exactly 4 specific and focused queries related to recent M&A news, announcements, deals, or trends. Ensure these queries help in finding up-to-date and fact-based information about the topic mentioned by the user.
Output 4 queries:
"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)
    generate_queries = (prompt_rag_fusion
                        | ChatOpenAI(model='o1-mini')
                        | StrOutputParser()
                        | (lambda x: x.split("\n")))
    queries = generate_queries.invoke({"question": question})
    print("[LOG] Requêtes générées :", queries)
    
    results = [retriever.invoke(q) for q in queries]
    print("[LOG] Documents récupérés :", results)
    
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            doc_str = dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)
    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [loads(d_str)]
    ]
    print(f"[LOG] Documents fusionnés : {len(reranked_docs)} documents rerankés.")
    
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])
    
    answer_template = """You are a financial journalist and M&A expert focusing on recent news. Using the following context extracted from M&A news sources, answer the user's question factually and succinctly. Highlight relevant and recent deals, events, or trends mentioned in the context. Do not invent information not provided in the context. Always give your source. It's in the name of the document you are using the information (it's Arx or CFNews. Do not say that it comes from the context, always say that it is from Arx or CFNews from the name of the doc in the metadata).

Context:
{context}

Question: {question}

If you are asked to do a market sheet or a company profile, structure the answer you give me based on the context.
Else, provide a clear and fact-based answer drawn from the provided context.
"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')
    final_input = {"context": context, "question": question}
    answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
    
    print("[LOG] Réponse actualités générée.")
    return answer


# --- Fonction RAG Fusion Fonds ---
def rag_fusion_fonds(question: str) -> str:
    print("[LOG] Démarrage de rag_fusion_fonds pour la question :", question)
    local_index_dir = "./Data/FAISS_index_fonds"
    local_index_file = os.path.join(local_index_dir, "index.faiss")
    gcs_blob_path = "FAISS_index_fonds/index.faiss"
    if not os.path.exists(local_index_file):
        download_file_from_gcs(GCS_BUCKET, gcs_blob_path, local_index_file)
    else:
        print(f"[LOG] Fichier index fonds déjà présent : {local_index_file}")
    
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(local_index_dir, embeddings=embedding, allow_dangerous_deserialization=True)
    print("[LOG] Index fonds chargé.")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20, "score_threshold": 0.01})
    
    query_generation_template = """You are an expert in private equity and investment funds. The user has asked a question related to investment funds, their strategies, sectors, geographic focus, or recent deals. Given the user's query:

{question}

Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)
    generate_queries = (prompt_rag_fusion
                        | ChatOpenAI(model='o1-mini')
                        | StrOutputParser()
                        | (lambda x: x.split("\n")))
    queries = generate_queries.invoke({"question": question})
    print("[LOG] Requêtes générées :", queries)
    
    results = [retriever.invoke(q) for q in queries]
    print("[LOG] Documents récupérés :", results)
    
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            doc_str = dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)
    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [loads(d_str)]
    ]
    print(f"[LOG] Documents fusionnés : {len(reranked_docs)} documents rerankés.")
    
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])
    
    answer_template = """You are a private equity and investment fund specialist. Using the following context sourced from a database of investment funds, answer the user's question with a focus on fund characteristics such as ticket size, sector preferences, geographic focus, and investment strategies. Provide a factual, clear, and concise explanation based solely on the provided information. Do not invent details not found in the context. Always give your source. It's in the name of the document you are using the information (it's Arx or CFNews. Do not say that it comes from the context, always say that it is from Arx or CFNews from the name of the doc in the metadata).

Context:
{context}

Question: {question}

Offer a fact-based, to-the-point response, highlighting key investment criteria or fund details.
"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')
    final_input = {"context": context, "question": question}
    answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
    
    print("[LOG] Réponse fonds générée.")
    return answer


# --- Fonction RAG Fusion Fiche Société vers Word ---
def rag_fusion_fiche_societe_to_word(question: str) -> dict:
    """
    Interroge la base d'actualités M&A via RAG et retourne une réponse structurée adaptée pour remplir un template Word.
    """
    print("[LOG] Démarrage de rag_fusion_fiche_societe_to_word pour la question :", question)
    local_index_dir = "./Data/FAISS_index_actualites"
    local_index_file = os.path.join(local_index_dir, "index.faiss")
    gcs_blob_path = "FAISS_index_actualites/index.faiss"
    if not os.path.exists(local_index_file):
        download_file_from_gcs(GCS_BUCKET, gcs_blob_path, local_index_file)
    else:
        print(f"[LOG] Fichier index actualités déjà présent : {local_index_file}")
    
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(local_index_dir, embeddings=embedding, allow_dangerous_deserialization=True)
    print("[LOG] Index actualités chargé pour fiche société.")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "score_threshold": 0.01})
    
    query_generation_template = """You are a knowledgeable M&A news analyst. Your role is to generate multiple targeted search queries to retrieve the most relevant and recent M&A news from a specialized news database. Always give your source. It's in the name of the document you are using the information (it's Arx or CFNews. Do not say that it comes from the context, always say that it is from Arx or CFNews from the name of the doc in the metadata).

Given the user's question: {question}

Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)
    generate_queries = (prompt_rag_fusion
                        | ChatOpenAI(model='o1-mini')
                        | StrOutputParser()
                        | (lambda x: x.split("\n")))
    queries = generate_queries.invoke({"question": question})
    print("[LOG] Requêtes générées :", queries)
    
    results = [retriever.invoke(q) for q in queries]
    print("[LOG] Documents récupérés :", results)
    
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            doc_str = json.dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)
    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [json.loads(d_str)]
    ]
    print(f"[LOG] Documents fusionnés : {len(reranked_docs)} documents rerankés.")
    
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])
    
    answer_template = """You are a financial journalist and M&A expert. You MUST answer in JSON format only, strictly matching the provided structure. Always give your source. It's in the name of the document you are using the information (it's Arx or CFNews. Do not say that it comes from the context, always say that it is from Arx or CFNews from the name of the doc in the metadata).

Context:
{context}

Question: {question}

Respond ONLY inside the following structure (do not include any explanations, formatting notes, or extra text outside the brackets). The first character of your answer must be a bracket and the last one too. It's VERY IMPORTANT THAT NOTHING IS OUTSIDE THE BRACKETS. DON'T WRITE '''json at the beginning, start immediately with the company_name !! :

{{
    "nom_societe": "Provide the company name if found",
    "description_activite": "Provide detailed description of the company's activities. Write it like you will present it, it needs to be 5-7 lines long with a good expression.",
    "chiffres_cles": "Include key metrics such as revenue, employees count, or founding date",
    "clients_par_secteur": "List the main clients of the company by sector",
    "implantation_positionnement": "List cities or countries where the company is located",
    "elements_financiers": "Describe the financial growth over the past 3 years",
    "president": "Name of the president",
    "daf": "Name of the financial director",
    "actionnaire": "List shareholders or investment funds associated with the company",
    "actionnaire_pourcentage": "Shareholder distribution percentages if available or majority/minority if available",
    "creanciers_type": "Type of creditors",
    "creanciers_commentaires": "Comments about the creditors",
    "actualites_presse": "Give as much information as possible (that you have)",
    "equity_story": "Provide details of investments or equity-related events. Give as much detail as you can. Present and order it for your client!",
    "creation": "Provide creation details or year of founding",
    "acquisition": "Give as much information as possible (that you have)"
}}
"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')
    final_input = {"context": context, "question": question}
    answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
    
    try:
        print("[LOG] Réponse du LLM:", answer)
        answer_dict = json.loads(answer)
    except json.JSONDecodeError:
        answer_dict = {}
        print("[LOG] Erreur lors du parsing JSON de la réponse du LLM.")
    
    return answer_dict


# --- Fonction RAG Fusion Multiples Transactions Comparables ---
def rag_fusion_multiples_transactions_comparables(question: str) -> str:
    print("[LOG] Démarrage de rag_fusion_multiples_transactions_comparables pour la question :", question)
    local_index_dir = "./Data/FAISS_index_multiples"
    local_index_file = os.path.join(local_index_dir, "index.faiss")
    gcs_blob_path = "FAISS_index_multiples/index.faiss"
    if not os.path.exists(local_index_file):
        download_file_from_gcs(GCS_BUCKET, gcs_blob_path, local_index_file)
    else:
        print(f"[LOG] Fichier index multiples déjà présent : {local_index_file}")
    
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(local_index_dir, embeddings=embedding, allow_dangerous_deserialization=True)
    print("[LOG] Index multiples chargé.")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20, "score_threshold": 0.01})
    
    query_generation_template = """You are an expert in mergers, acquisitions, and financial transactions. 
The user has asked a question related to comparable transactions or multiples such as EV/Revenue, EV/EBITDA, or EV/EBIT. 
Given the user's query:

{question}

Your task is to generate five alternative versions of the user question to retrieve relevant documents from a vector database. The goal is to capture different perspectives on the user's query to ensure relevant documents are retrieved. Provide these alternative questions separated by newlines.
"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)
    generate_queries = (prompt_rag_fusion 
                        | ChatOpenAI(model='o1-mini') 
                        | StrOutputParser() 
                        | (lambda x: x.split("\n")))
    queries = generate_queries.invoke({"question": question})
    print("[LOG] Requêtes générées :", queries)
    
    results = [retriever.invoke(q) for q in queries]
    print("[LOG] Documents récupérés :", results)
    
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            doc_str = dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)
    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [loads(d_str)]
    ]
    print(f"[LOG] Documents fusionnés : {len(reranked_docs)} documents rerankés.")
    
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])
    
    answer_template = """You are an expert in financial transactions and valuation multiples. Using the following context sourced from a database of comparable transactions, provide insights about relevant transactions, key valuation multiples (e.g., EV/Revenue, EV/EBITDA), and deal characteristics.

Context:
{context}

Question: {question}

Your response should be factual, concise, and focused solely on the provided context. Include specific multiples, transaction details, and other financial metrics when possible. Always indicate the source of your information (it will always be MergerMarket.)
"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')
    final_input = {"context": context, "question": question}
    answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
    
    print("[LOG] Réponse multiples transactions générée.")
    return answer


import io
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfbase import pdfmetrics

def wrap_text(text, font_name, font_size, max_width):
    """
    Découpe le texte en lignes sans couper de mot, en respectant une largeur maximale.
    """
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if current_line == "":
            new_line = word
        else:
            new_line = current_line + " " + word
        # Si la largeur de la nouvelle ligne est inférieure ou égale à max_width, on l'ajoute
        if pdfmetrics.stringWidth(new_line, font_name, font_size) <= max_width:
            current_line = new_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def add_watermark_to_pdf(input_pdf_bytes: bytes, bank_name: str) -> bytes:
    """
    Ajoute un filigrane au PDF contenu dans input_pdf_bytes.
    Le filigrane aura la forme "Confidentiel - <bank_name>" (avec un espace après le tiret),
    et sera automatiquement mis en forme pour aller à la ligne si nécessaire, sans couper de mot,
    puis centré en fonction des dimensions de la première page du PDF.
    Retourne le PDF filigrané sous forme d'octets.
    """
    # Texte complet du filigrane
    watermark_text = f"Confidentiel - {bank_name.strip()}"
    
    # Lecture du PDF d'entrée pour obtenir les dimensions de la première page
    input_pdf_stream = io.BytesIO(input_pdf_bytes)
    reader = PdfReader(input_pdf_stream)
    if len(reader.pages) == 0:
        raise ValueError("Le PDF d'entrée ne contient aucune page.")
    first_page = reader.pages[0]
    page_width = float(first_page.mediabox.width)
    page_height = float(first_page.mediabox.height)
    
    # On définit le canvas en mode paysage (si le PDF est en format paysage, sinon, on utilise les dimensions de la page)
    # Ici, on utilisera les dimensions de la première page telles quelles.
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(page_width, page_height))
    
    # Définir la police et la taille
    font_name = "Helvetica"
    font_size = 40
    can.setFont(font_name, font_size)
    
    # Définir la largeur maximale pour le texte (80% de la largeur de la page)
    max_width = page_width * 0.8
    
    # Découper le texte en lignes si nécessaire
    lines = wrap_text(watermark_text, font_name, font_size, max_width)
    num_lines = len(lines)
    line_height = font_size * 1.2  # Espace entre les lignes
    # Calculer le décalage vertical pour centrer le bloc de texte
    total_text_height = num_lines * line_height
    start_y = total_text_height / 2 - line_height/2  # Ajustement pour centrer verticalement
    
    # Déplacer l'origine au centre de la page
    can.translate(page_width/2, page_height/2)
    # Appliquer une rotation de 45°
    can.rotate(45)
    can.setFillColorRGB(0, 0, 0, alpha=0.15)
    
    # Dessiner chaque ligne de texte centrée
    for i, line in enumerate(lines):
        line_width = pdfmetrics.stringWidth(line, font_name, font_size)
        x = -line_width / 2  # Centrage horizontal
        y = start_y - i * line_height
        can.drawString(x, y, line)
    
    can.save()
    
    packet.seek(0)
    watermark_pdf = PdfReader(packet)
    watermark_page = watermark_pdf.pages[0]
    
    # Fusionner le filigrane sur chaque page
    writer = PdfWriter()
    for page in reader.pages:
        page.merge_page(watermark_page)
        writer.add_page(page)
    
    output_pdf_stream = io.BytesIO()
    writer.write(output_pdf_stream)
    output_pdf_stream.seek(0)
    
    print("[LOG] Filigranage terminé.", flush=True)
    return output_pdf_stream.getvalue()