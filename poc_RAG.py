#!/usr/bin/env python
# coding: utf-8

import os
import json
from json import dumps, loads
from operator import itemgetter
import datetime
import pandas as pd
import bs4
import time
import random
import requests  # Pour télécharger depuis GitHub

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

# --- Configuration des variables d'environnement ---
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'
# On récupère la clé OPENAI_API_KEY depuis l'environnement
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
    raise ValueError("La variable d'environnement OPENAI_API_KEY n'est pas définie. Veuillez la définir dans vos secrets.")
os.environ['OPENAI_API_KEY'] = openai_api_key

# --- Téléchargement depuis GitHub ---
# Définissez l'URL de base de votre dépôt public GitHub
GITHUB_BASE_URL = "https://github.com/Noeamar/RAG_MnA/tree/main"

def download_file_from_github(source_blob_name: str, destination_file_name: str):
    """
    Télécharge un fichier depuis GitHub via son URL brute et le sauvegarde localement.
    
    :param source_blob_name: Chemin relatif dans le repo (ex: "FAISS_index/index.faiss")
    :param destination_file_name: Chemin local où sauvegarder le fichier.
    """
    url = GITHUB_BASE_URL + source_blob_name
    print(f"[LOG] Téléchargement de {url} vers {destination_file_name}...", flush=True)
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"[LOG] Erreur lors du téléchargement : code {response.status_code}")
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    with open(destination_file_name, "wb") as f:
        f.write(response.content)
    os.chmod(destination_file_name, 0o777)
    file_size = os.stat(destination_file_name).st_size
    print(f"[LOG] Téléchargement terminé. Taille du fichier: {file_size} octets", flush=True)
    if file_size == 0:
        raise ValueError(f"[LOG] Le fichier {destination_file_name} est vide après téléchargement.")

# --- Fonctions RAG Fusion (sans lien avec des graphes) ---
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def rag_fusion(question: str) -> str:
    print("[LOG] Démarrage de rag_fusion pour la question :", question, flush=True)
    # Définir le répertoire et le chemin du fichier local
    local_index_dir = "./Data/FAISS_index"
    local_index_file = os.path.join(local_index_dir, "index.faiss")
    github_file_path = "index.faiss"  # Chemin relatif dans votre repo GitHub
    
    # Forcer la suppression du fichier local s'il existe pour forcer le téléchargement
    if os.path.exists(local_index_file):
        os.remove(local_index_file)
        print("[LOG] Fichier existant supprimé pour forcer le téléchargement depuis GitHub.", flush=True)
    
    # Télécharger le fichier depuis GitHub
    download_file_from_github(github_file_path, local_index_file)
    
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(local_index_dir, embeddings=embedding, allow_dangerous_deserialization=True)
    print("[LOG] Index FAISS chargé.", flush=True)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "score_threshold": 0.01})
    
    query_generation_template = """You are a seasoned M&A consultant with access to a broad dataset that includes recent news, company profiles, and investment fund details. Given the user's question:

{question}

Generate exactly 4 focused queries that will help retrieve the most relevant and comprehensive information from this rich dataset.
"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)
    generate_queries = (prompt_rag_fusion
                        | ChatOpenAI(model='o1-mini')
                        | StrOutputParser()
                        | (lambda x: x.split("\n")))
    queries = generate_queries.invoke({"question": question})
    print("[LOG] Requêtes générées :", queries, flush=True)
    
    results = [retriever.invoke(q) for q in queries]
    print("[LOG] Documents récupérés :", results, flush=True)
    
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
    print(f"[LOG] Documents fusionnés : {len(reranked_docs)} documents rerankés.", flush=True)
    
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])
    
    answer_template = """You are an M&A expert who can analyze recent news, company data, and fund information to provide a comprehensive and accurate answer. Using the following context extracted from various M&A-related sources (news, companies, funds), answer the user's question concisely and factually. Highlight relevant deals, company details, or fund strategies if mentioned. Do not invent information that isn't provided. Always give your source.

Context:
{context}

Question: {question}

Provide a clear, fact-based answer focusing on the M&A domain.
"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')
    final_input = {"context": context, "question": question}
    answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
    
    print("[LOG] Réponse générée.", flush=True)
    return answer

def rag_fusion_actualites(question: str) -> str:
    print("[LOG] Démarrage de rag_fusion_actualites pour la question :", question)
    
    # Définissez ici la liste des dossiers batch contenant les index FAISS
    batch_dirs = [
        "./Data/FAISS_index_actualites_NLP_400_0_batch_1",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_2",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_3"
    ]
    
    # Vous pouvez ajouter ici du code pour télécharger chaque batch depuis GitHub s'il manque
    for batch in batch_dirs:
        if not os.path.exists(batch):
            print(f"[LOG] Attention, le dossier {batch} n'existe pas.")
            # download_file_from_github(...)  # Vous pouvez adapter cette partie si besoin.
    
    embedding = OpenAIEmbeddings()
    retrievers = []
    
    # Charger chaque index FAISS et créer un retriever avec le nouveau paramétrage
    for batch in batch_dirs:
        print(f"[LOG] Chargement de l'index depuis {batch}")
        vectorstore = FAISS.load_local(batch, embeddings=embedding, allow_dangerous_deserialization=True)
        # Utilisation du retriever en mode "similarity_score_threshold"
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.6, "k": 10}
        )
        retrievers.append(retriever)
    
    print("[LOG] Tous les index batch sont chargés.")
    
    # Génération des requêtes via le prompt
    query_generation_template = ("You are a helpful assistant that generates multiple search queries based on a single input query. \n"
    "Generate multiple search queries related to: {question} \n"
    "Output (3 queries):"
    )
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)
    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(model='o1-mini')
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    queries = generate_queries.invoke({"question": question})
    print("[LOG] Requêtes générées :", queries)
    
    # Pour chaque requête, interroger tous les retrievers et combiner les résultats
    all_results = []
    for q in queries:
        query_results = []
        for retriever in retrievers:
            # Utilisez .invoke(q) pour récupérer les documents pour la requête q
            docs = retriever.invoke(q)
            query_results.extend(docs)
        all_results.append(query_results)
    print("[LOG] Documents récupérés pour chaque requête.")
    
    # Fusion des résultats avec Reciprocal Rank Fusion (RRF)
    fused_scores = {}
    for docs in all_results:
        for rank, doc in enumerate(docs):
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            doc_str = json.dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 100)
    
    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [json.loads(d_str)]
    ]
    print(f"[LOG] Documents fusionnés : {len(reranked_docs)} documents rerankés.")
    
    # Construire le contexte à partir des documents rerankés
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])
    
    # Génération de la réponse finale avec le prompt de réponse
    answer_template = (
        "You are a financial journalist and M&A expert focusing on recent news. Using the following context "
        "extracted from M&A news sources, answer the user's question."
        "Always present the information chronologically."
        "Always give your source. It's always given in the title of the file the information is extracted from. It's either Arx or CFNews.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
    )
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')
    final_input = {"context": context, "question": question}
    answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
    
    print("[LOG] Réponse générée.")
    return answer

    
def rag_fusion_fonds(question: str) -> str:
    print("[LOG] Démarrage de rag_fusion_fonds pour la question :", question)
    local_index_dir = "./Data/FAISS_index_fonds"
    local_index_file = os.path.join(local_index_dir, "index.faiss")
    github_file_path = "FAISS_index_fonds/index.faiss"
    if not os.path.exists(local_index_file):
        download_file_from_github(github_file_path, local_index_file)
    else:
        print(f"[LOG] Fichier index fonds déjà présent : {local_index_file}")
    
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(local_index_dir, embeddings=embedding, allow_dangerous_deserialization=True)
    print("[LOG] Index fonds chargé.")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20, "score_threshold": 0.01})
    
    query_generation_template = """You are an expert in private equity and investment funds. Given the user's query:

{question}

Generate five alternative versions of the question to retrieve relevant documents from a vector database.
Provide these alternative questions separated by newlines.
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
    
    answer_template = """You are a private equity and investment fund specialist. Using the following context sourced from a database of investment funds, answer the user's question focusing on fund characteristics such as ticket size, sector preferences, geographic focus, and investment strategies. Provide a factual and concise explanation based solely on the provided information. Always indicate the source (Arx).

Context:
{context}

Question: {question}

Offer a fact-based response highlighting key investment criteria.
"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')
    final_input = {"context": context, "question": question}
    answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
    
    print("[LOG] Réponse fonds générée.")
    return answer

def rag_fusion_fiche_societe_to_word(question: str) -> dict:
    """
    Interroge la base d'actualités M&A via RAG et retourne une réponse structurée pour remplir un template Word.
    """
    print("[LOG] Démarrage de rag_fusion_fiche_societe_to_word pour la question :", question)
    # Définissez ici la liste des dossiers batch contenant les index FAISS
    batch_dirs = [
        "./Data/FAISS_index_actualites_NLP_400_0_batch_1",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_2",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_3"
    ]
    
    # Vous pouvez ajouter ici du code pour télécharger chaque batch depuis GitHub s'il manque
    for batch in batch_dirs:
        if not os.path.exists(batch):
            print(f"[LOG] Attention, le dossier {batch} n'existe pas.")
            # download_file_from_github(...)  # Vous pouvez adapter cette partie si besoin.
    
    embedding = OpenAIEmbeddings()
    retrievers = []
    
    # Charger chaque index FAISS et créer un retriever avec le nouveau paramétrage
    for batch in batch_dirs:
        print(f"[LOG] Chargement de l'index depuis {batch}")
        vectorstore = FAISS.load_local(batch, embeddings=embedding, allow_dangerous_deserialization=True)
        # Utilisation du retriever en mode "similarity_score_threshold"
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.6, "k": 10}
        )
        retrievers.append(retriever)
    
    print("[LOG] Tous les index batch sont chargés.")
    
    # Génération des requêtes via le prompt
    query_generation_template = ("You are a helpful assistant that generates multiple search queries based on a single input query. \n"
    "Generate multiple search queries related to: {question} \n"
    "Output (3 queries):"
    )
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)
    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(model='o1-mini')
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    queries = generate_queries.invoke({"question": question})
    print("[LOG] Requêtes générées :", queries)
    
    # Pour chaque requête, interroger tous les retrievers et combiner les résultats
    all_results = []
    for q in queries:
        query_results = []
        for retriever in retrievers:
            # Utilisez .invoke(q) pour récupérer les documents pour la requête q
            docs = retriever.invoke(q)
            query_results.extend(docs)
        all_results.append(query_results)
    print("[LOG] Documents récupérés pour chaque requête.")
    
    # Fusion des résultats avec Reciprocal Rank Fusion (RRF)
    fused_scores = {}
    for docs in all_results:
        for rank, doc in enumerate(docs):
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            doc_str = json.dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 100)
    
    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [json.loads(d_str)]
    ]
    print(f"[LOG] Documents fusionnés : {len(reranked_docs)} documents rerankés.")
    
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])
    
    answer_template = """You are a financial journalist and M&A expert. You MUST answer in JSON format only, strictly matching the provided structure.

Context:
{context}

Question: {question}

Rewrite everything you are given in the context so that there are nice sentences.

Respond ONLY within the following structure (no extra text):

{{
    "nom_societe": "Provide the company name if found",
    "description_activite": "Provide a detailed description of the company's activities (5-10 lines), using clear and concise language.",
    "chiffres_cles": "Include key metrics such as revenue, employee count, or founding date, summarized if needed.",
    "clients_par_secteur": "List the main clients by sector and give their name.",
    "implantation_positionnement": "List cities or countries where the company is located",
    "elements_financiers": "Summarize the financial growth over the past 3 years in a concise manner",
    "president": "Name of the president",
    "daf": "Name of the financial director",
    "actionnaire": "Provide a summarized list of key shareholders or investment funds, with the most important ones highlighted",
    "actionnaire_pourcentage": "Shareholder distribution percentages if available",
    "creanciers_type": "List types of creditors concisely",
    "creanciers_commentaires": "Provide a brief summary of the creditors' comments",
    "actualites_presse": "Present with maximum details and a clear language recent press news in a clear format",
    "equity_story": "Present with maximum details and a clear language major equity events and investments",
    "creation": "Present with maximum details and a clear language creation details or year of founding",
    "acquisitions": "Present with maximum details and a clear language all acquisitions, including dates and a descriptions"
}}
"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')
    final_input = {"context": context, "question": question}
    answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
    
    try:
        raw_answer = answer.strip()
        # Retirer les balises markdown si présentes
        if raw_answer.startswith("```json"):
            raw_answer = raw_answer[len("```json"):].strip()
        if raw_answer.endswith("```"):
            raw_answer = raw_answer[:-3].strip()
        answer_dict = json.loads(raw_answer)
    except json.JSONDecodeError as e:
        answer_dict = {}
        print("[LOG] Erreur lors du parsing JSON de la réponse du LLM:", e)

    return answer_dict

def rag_fusion_fiche_societe_to_word_websearch(question: str) -> dict:
    print("[LOG] Démarrage de rag_fusion_fiche_societe_to_word_websearch pour la question :", question)
    
    # Liste des dossiers batch contenant les index FAISS
    batch_dirs = [
        "./Data/FAISS_index_actualites_NLP_400_0_batch_1",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_2",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_3"
    ]
    
    # Vérifier que chaque batch existe (téléchargement depuis GitHub si nécessaire)
    for batch in batch_dirs:
        if not os.path.exists(batch):
            print(f"[LOG] Attention, le dossier {batch} n'existe pas.")
            # download_file_from_github(...)  # À implémenter si besoin.
    
    embedding = OpenAIEmbeddings()
    retrievers = []
    
    # Charger chaque index FAISS et créer un retriever en mode "similarity_score_threshold"
    for batch in batch_dirs:
        print(f"[LOG] Chargement de l'index depuis {batch}")
        vectorstore = FAISS.load_local(batch, embeddings=embedding, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.6, "k": 10}
        )
        retrievers.append(retriever)
    
    print("[LOG] Tous les index batch sont chargés.")
    
    # Génération des requêtes via le prompt
    query_generation_template = (
        "You are a helpful assistant that generates multiple search queries based on a single input query. \n"
        "Generate multiple search queries related to: {question} \n"
        "Output (3 queries):"
    )
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)
    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(model='o1-mini')
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    queries = generate_queries.invoke({"question": question})
    print("[LOG] Requêtes générées :", queries)
    
    # Pour chaque requête, interroger tous les retrievers et combiner les résultats
    all_results = []
    for q in queries:
        query_results = []
        for retriever in retrievers:
            docs = retriever.invoke(q)
            query_results.extend(docs)
        all_results.append(query_results)
    print("[LOG] Documents récupérés pour chaque requête.")
    
    # Fusionner les résultats avec Reciprocal Rank Fusion (RRF)
    fused_scores = {}
    for docs in all_results:
        for rank, doc in enumerate(docs):
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            doc_str = json.dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 100)
    
    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [json.loads(d_str)]
    ]
    print(f"[LOG] Documents fusionnés : {len(reranked_docs)} documents rerankés.")
    
    # Pour alléger le contexte, ne conserver que les 10 premiers documents
    selected_docs = reranked_docs[:50]
    context = "\n\n".join([doc.page_content for doc, _ in selected_docs])
    
    # Prompt final avec instruction explicite sur l'utilisation du contexte
    answer_template = """You are a financial journalist and M&A expert. You MUST answer in JSON format only, strictly matching the provided structure.

Use primarily the context provided below (abridged) to construct your answer. If the context lacks certain details, supplement your answer with the most relevant results from your internet search.

Context (abridged):
{context}

Question: {question}.

Respond ONLY within the following JSON structure (no extra text):

{{
    "nom_societe": "Provide the company name if found",
    "description_activite": "Provide a detailed description of the company's activities (5-10 lines), using clear and concise language.",
    "chiffres_cles": "Include key metrics such as revenue, employee count, or founding date, summarized if needed.",
    "clients_par_secteur": "List the main clients by sector and give their names.",
    "implantation_positionnement": "List cities or countries where the company is located.",
    "elements_financiers": "Summarize the financial growth over the past 3 years in a concise manner.",
    "president": "Name of the president",
    "daf": "Name of the financial director",
    "actionnaire": "Provide a summarized list of key shareholders or investment funds, with the most important ones highlighted.",
    "actionnaire_pourcentage": "Shareholder distribution percentages if available.",
    "creanciers_type": "List types of creditors concisely.",
    "creanciers_commentaires": "Provide a brief summary of the creditors' comments.",
    "actualites_presse": "Present recent press news with maximum details and clear language.",
    "equity_story": "Present major equity events and investments (e.g., LBO, MBO) with maximum details and clear language.",
    "creation": "Present the company's creation details or founding year with maximum details and clear language.",
    "acquisitions": "Present all key acquisitions, build-ups, mergers with dates and descriptions with maximum details and clear language."
}}
"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='gpt-4o-mini-search-preview', model_kwargs={"web_search_options": {}})
    final_input = {"context": context, "question": question}
    
    max_attempts = 3
    answer_dict = {}
    for attempt in range(max_attempts):
        print(f"[LOG] Tentative {attempt+1} pour générer la réponse.")
        answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
        raw_answer = answer.strip()
        print(f"[DEBUG] Réponse brute générée (tentative {attempt+1}) : {raw_answer}")
        if raw_answer.startswith("```json"):
            raw_answer = raw_answer[len("```json"):].strip()
        if raw_answer.endswith("```"):
            raw_answer = raw_answer[:-3].strip()
        try:
            answer_dict = json.loads(raw_answer)
            print("[LOG] Réponse JSON générée avec succès.")
            break
        except json.JSONDecodeError as e:
            print("[LOG] Erreur lors du parsing JSON de la réponse du LLM:", e)
            if attempt == max_attempts - 1:
                print("[LOG] Nombre maximum de tentatives atteint, aucune réponse valide obtenue.")
                answer_dict = {}
            else:
                sleep(2)
    
    return answer_dict

# def internet_search(query: str) -> list:
#     """
#     Effectue une recherche Internet avec le modèle gpt-4o-mini-search-preview
#     et renvoie directement les 50 meilleurs résultats sous forme de Documents.
#     """
#     search_prompt = (
#         f"Search the internet for the following query and provide the top 50 results "
#         f"in JSON format as a list of objects with keys 'title' and 'snippet'. It is imperative that the results have this form ! : {query}"
#     )
#     llm_internet = ChatOpenAI(model='gpt-4o-mini-search-preview', model_kwargs={"web_search_options": {}})
#     response = llm_internet.invoke(search_prompt)
#     # Convertir la réponse en chaîne si nécessaire
#     if not isinstance(response, str):
#         response = str(response)
#     try:
#         results = json.loads(response)
#         docs = []
#         for item in results:
#             content = f"Title: {item.get('title', '')}\nSnippet: {item.get('snippet', '')}"
#             docs.append(Document(page_content=content, metadata={}))
#         return docs[:50]
#     except Exception as e:
#         print("[LOG] Erreur lors du parsing des résultats Internet pour la requête:", query, e)
#         return []


# def rag_fusion_fiche_societe_to_word_websearch(question: str) -> dict:
#     print("[LOG] Démarrage de rag_fusion_fiche_societe_to_word_websearch pour la question :", question)
    
#     # --- Définition des dossiers batch locaux ---
#     batch_dirs = [
#         "./Data/FAISS_index_actualites_NLP_400_0_batch_1",
#         "./Data/FAISS_index_actualites_NLP_400_0_batch_2",
#         "./Data/FAISS_index_actualites_NLP_400_0_batch_3"
#     ]
#     for batch in batch_dirs:
#         if not os.path.exists(batch):
#             print(f"[LOG] Attention, le dossier {batch} n'existe pas.")
#             # download_file_from_github(...)  # À adapter si nécessaire.
    
#     # --- Multi Query Generation ---
#     query_generation_template = (
#         "Vous êtes un expert en actualités M&A. À partir de la question suivante : {question}\n"
#         "Générez exactement 3 requêtes alternatives spécifiques pour rechercher des actualités M&A pertinentes.\n"
#         "Fournissez ces 3 requêtes séparées par des sauts de ligne :"
#     )
#     prompt_query = ChatPromptTemplate.from_template(query_generation_template)
#     generate_queries_chain = (
#         prompt_query
#         | ChatOpenAI(model='o1-mini')
#         | StrOutputParser()
#         | (lambda x: x.split("\n"))
#     )
#     queries = generate_queries_chain.invoke({"question": question})
#     # Nettoyer la liste pour supprimer les marqueurs Markdown et les lignes vides
#     queries = [q.strip() for q in queries if q.strip() and q.strip() != "```"]
#     print("[LOG] Requêtes nettoyées :", queries)
    
#     # --- Recherche locale sur les 3 batchs ---
#     local_results = []
#     embedding = OpenAIEmbeddings()
#     for batch in batch_dirs:
#         print(f"[LOG] Chargement de l'index depuis {batch}")
#         vectorstore = FAISS.load_local(batch, embeddings=embedding, allow_dangerous_deserialization=True)
#         retriever = vectorstore.as_retriever(
#             search_type="similarity_score_threshold",
#             search_kwargs={"score_threshold": 0.6, "k": 15}
#         )
#         for q in queries:
#             results = retriever.invoke(q)
#             local_results.extend(results)
#     print(f"[LOG] Recherche locale totale : {len(local_results)} documents récupérés.")
    
#     # --- Fusion RRF sur les résultats locaux ---
#     fused_scores = {}
#     for rank, doc in enumerate(local_results):
#         doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
#         doc_str = json.dumps(doc_dict)
#         if doc_str not in fused_scores:
#             fused_scores[doc_str] = 0
#         fused_scores[doc_str] += 1 / (rank + 60)
#     reranked_local_docs = [
#         (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
#         for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
#         for d in [json.loads(d_str)]
#     ]
#     print(f"[LOG] Documents locaux fusionnés (RRF) : {len(reranked_local_docs)} documents rerankés.")
    
#     # --- Recherche Internet (directe) ---
#     internet_results = []
#     print(queries)
#     for q in queries:
#         print(f"[LOG] Recherche Internet pour la requête : {q}")
#         docs = internet_search(q)
#         print(f"[LOG] {len(docs)} documents trouvés sur Internet.")
#         internet_results.extend(docs)
#     internet_results = internet_results[:50]
#     print(f"[LOG] Recherche Internet : {len(internet_results)} documents récupérés.")
    
#     # --- Combinaison des résultats ---
#     combined_results = [doc for doc, _ in reranked_local_docs] + internet_results
#     print(f"[LOG] Total documents combinés (locaux + Internet) : {len(combined_results)} documents.")
    
#     # --- Construction du contexte final ---
#     context = "\n\n".join([doc.page_content for doc in combined_results])
    
#     # --- Prompt final ---
#     # Utilisez le template fourni
#     answer_template = """You are a financial journalist and M&A expert. You MUST answer in JSON format only, strictly matching the provided structure.

# Context:
# {context}

# Question: {question}

# Rewrite everything you are given in the context so that there are nice sentences.

# Respond ONLY within the following structure (no extra text):

# {{
#     "nom_societe": "Provide the company name if found",
#     "description_activite": "Provide a detailed description of the company's activities (5-10 lines), using clear and concise language.",
#     "chiffres_cles": "Include key metrics such as revenue, employee count, or founding date, summarized if needed.",
#     "clients_par_secteur": "List the main clients by sector and give their name.",
#     "implantation_positionnement": "List cities or countries where the company is located",
#     "elements_financiers": "Summarize the financial growth over the past 3 years in a concise manner",
#     "president": "Name of the president",
#     "daf": "Name of the financial director",
#     "actionnaire": "Provide a summarized list of key shareholders or investment funds, with the most important ones highlighted",
#     "actionnaire_pourcentage": "Shareholder distribution percentages if available",
#     "creanciers_type": "List types of creditors concisely",
#     "creanciers_commentaires": "Provide a brief summary of the creditors' comments",
#     "actualites_presse": "Present with maximum details and a clear language recent press news in a clear format",
#     "equity_story": "Present with maximum details and a clear language major equity events and investments",
#     "creation": "Present with maximum details and a clear language creation details or year of founding",
#     "acquisitions": "Present with maximum details and a clear language all acquisitions, including dates and a descriptions"
# }}
# """
#     answer_prompt = ChatPromptTemplate.from_template(answer_template)
#     formatted_prompt = answer_prompt.format(context=context, question=question)
    
#     llm_final = ChatOpenAI(model='o1-mini')
    
#     max_attempts = 3
#     answer_dict = {}
#     for attempt in range(max_attempts):
#         print(f"[LOG] Tentative {attempt+1} pour générer la réponse finale.")
#         answer = (llm_final | StrOutputParser()).invoke(formatted_prompt)
#         raw_answer = answer.strip()
#         print(f"[DEBUG] Réponse brute générée (tentative {attempt+1}) : {raw_answer}")
#         if raw_answer.startswith("```json"):
#             raw_answer = raw_answer[len("```json"):].strip()
#         if raw_answer.endswith("```"):
#             raw_answer = raw_answer[:-3].strip()
#         try:
#             answer_dict = json.loads(raw_answer)
#             print("[LOG] Réponse JSON générée avec succès.")
#             break
#         except json.JSONDecodeError as e:
#             print("[LOG] Erreur lors du parsing JSON de la réponse du LLM:", e)
#             if attempt == max_attempts - 1:
#                 print("[LOG] Nombre maximum de tentatives atteint, aucune réponse valide obtenue.")
#                 answer_dict = {}
#             else:
#                 sleep(2)
    
#     return answer_dict



def generate_fiche_societe(company_data: dict, template_path: str, output_path: str):
    """
    Remplit le template Word avec les données de l'entreprise et sauvegarde le document.

    :param company_data: Dictionnaire contenant les informations de la fiche (générées par le LLM)
    :param template_path: Chemin vers le template Word (local ou téléchargé depuis GitHub)
    :param output_path: Chemin de sortie pour sauvegarder la fiche remplie
    """
    doc = DocxTemplate(template_path)
    doc.render(company_data)
    doc.save(output_path)


def rag_fusion_multiples_transactions_comparables(question: str) -> str:
    print("[LOG] Démarrage de rag_fusion_multiples_transactions_comparables pour la question :", question)
    local_index_dir = "./Data/FAISS_index_multiples"
    local_index_file = os.path.join(local_index_dir, "index.faiss")
    github_file_path = "FAISS_index_multiples/index.faiss"
    if not os.path.exists(local_index_file):
        download_file_from_github(github_file_path, local_index_file)
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

Generate five alternative versions of the user question to retrieve relevant documents.
Provide these alternative questions separated by newlines.
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

Your response should be factual, concise, and focused solely on the provided context. Include specific multiples, transaction details, and other financial metrics when possible. Always indicate the source (MergerMarket).
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
    watermark_text = f"Confidentiel - {bank_name.strip()}"
    input_pdf_stream = io.BytesIO(input_pdf_bytes)
    reader = PdfReader(input_pdf_stream)
    if len(reader.pages) == 0:
        raise ValueError("Le PDF d'entrée ne contient aucune page.")
    first_page = reader.pages[0]
    page_width = float(first_page.mediabox.width)
    page_height = float(first_page.mediabox.height)
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(page_width, page_height))
    font_name = "Helvetica"
    font_size = 40
    can.setFont(font_name, font_size)
    max_width = page_width * 0.8
    lines = wrap_text(watermark_text, font_name, font_size, max_width)
    num_lines = len(lines)
    line_height = font_size * 1.2
    total_text_height = num_lines * line_height
    start_y = total_text_height / 2 - line_height/2
    can.translate(page_width/2, page_height/2)
    can.rotate(45)
    can.setFillColorRGB(0, 0, 0, alpha=0.15)
    for i, line in enumerate(lines):
        line_width = pdfmetrics.stringWidth(line, font_name, font_size)
        x = -line_width / 2
        y = start_y - i * line_height
        can.drawString(x, y, line)
    can.save()
    packet.seek(0)
    watermark_pdf = PdfReader(packet)
    watermark_page = watermark_pdf.pages[0]
    writer = PdfWriter()
    for page in reader.pages:
        page.merge_page(watermark_page)
        writer.add_page(page)
    output_pdf_stream = io.BytesIO()
    writer.write(output_pdf_stream)
    output_pdf_stream.seek(0)
    print("[LOG] Filigranage terminé.", flush=True)
    return output_pdf_stream.getvalue()
