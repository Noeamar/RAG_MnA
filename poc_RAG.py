#!/usr/bin/env python
# coding: utf-8

# In[28]:


import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
import pandas as pd
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
from langchain.load import dumps, loads
from operator import itemgetter
import os

# In[4]:


import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'

# In[5]:


openai_api_key = os.getenv("OPENAI_API_KEY")

# ### Remodel du fichier CSV Scraped Companies

# In[ ]:


# # Chemin du fichier d'entrée
# input_file_path = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_M-A\\Data\\scraped_companies_all_columns.csv"

# # Chemin du fichier de sortie
# output_file_path = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_M-A\\Data\\scraped_companies_reorganized.csv"

# # Charger le fichier CSV
# df = pd.read_csv(input_file_path)

# # Renommer les colonnes
# df.columns = [
#     "Logo", "Column_to_remove", "Société", "Pays Siege",
#     "Siège répertorié sur Arx", "Site internet", "Secteur",
#     "Mots-clés", "Description", "Dernier CA (M)", "Périmètre CA",
#     "Column_to_remove_2", "TCAM (%)", "Dirigeants/Equipe CF", "Cotation", "ISIN"
# ]


# # Sélectionner et réorganiser les colonnes selon la spécification
# columns_to_keep = [
#     "Société",
#     "Pays Siege",
#     "Siège répertorié sur Arx",
#     "Site internet",
#     "Secteur",
#     "Mots-clés",
#     "Description",
#     "Dernier CA (M)",
#     "Périmètre CA",
#     "TCAM (%)",
#     "Dirigeants/Equipe CF",
#     "Cotation",
#     "ISIN"
# ]

# # Renommer les colonnes sélectionnées pour plus de clarté (facultatif)
# renamed_columns = [
#     "Company", "Country Headquarters", "Arx Listed HQ", "Website",
#     "Sector", "Keywords", "Description", "Latest Revenue (M)",
#     "Revenue Scope", "CAGR (%)", "Executives/Team CF", "Rating", "ISIN"
# ]

# # Réorganiser et renommer les colonnes
# df_reorganized = df[columns_to_keep]
# df_reorganized.columns = renamed_columns

# # Enregistrer le fichier CSV modifié
# df_reorganized.to_csv(output_file_path, index=False, encoding="utf-8")

# print(f"Le fichier CSV a été réorganisé et enregistré sous : {output_file_path}")

# ### Chargement des fichiers dans docs pour préparer l'embedding

# In[ ]:


# # Fonction pour charger un fichier et extraire les données avec métadonnées
# def load_files_with_metadata(file_paths, metadata_file):
#     # Charger les métadonnées
#     with open(metadata_file, 'r') as meta_file:
#         metadata = json.load(meta_file)
    
#     docs = []
#     for path in file_paths:
#         file_name = path.split("\\")[-1]  # Extraire le nom du fichier
#         file_metadata = metadata.get(file_name, {})  # Récupérer les métadonnées

#         # Initialiser le contenu du document
#         content = ""
#         if path.endswith(".pdf"):
#             loader = PyPDFLoader(path)
#             content = "\n".join([doc.page_content for doc in loader.load()])
#         elif path.endswith(".txt"):
#             loader = TextLoader(path)
#             content = "\n".join([doc.page_content for doc in loader.load()])
#         elif path.endswith(".csv"):
#             df = pd.read_csv(path)
#             if 'Title' in df.columns and 'Content' in df.columns:
#                 df['text'] = df['Title'] + "\n\n" + df['Content']
#             else:
#                 df['text'] = df.apply(lambda row: ' '.join(map(str, row.values)), axis=1)
#             content = "\n".join(df['text'].tolist())
        
#         # Créer un document global pour le fichier avec les métadonnées
#         doc = Document(
#             page_content=content,
#             metadata={
#                 "file_name": file_name,
#                 **file_metadata  # Ajouter les métadonnées depuis le JSON
#             }
#         )
#         docs.append(doc)
    
#     return docs

# # Exemple d'utilisation
# file_paths = [
#     "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_M-A\\Data\\scraped_companies_reorganized.csv",
#     "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_M-A\\Data\\scraped_news_grid.csv",
#     "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_M-A\\Data\\exported_results.csv"
# ]

# metadata_file = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_M-A\\metadata_arx.json"

# # Charger les fichiers avec métadonnées
# docs = load_files_with_metadata(file_paths, metadata_file)

# # Résumé des documents chargés
# print(f"Nombre total de documents : {len(docs)}")
# for doc in docs:
#     print(f"Document : {doc.metadata['file_name']}, Métadonnées : {doc.metadata}")

# ### Embedding (Attention a ne pas le lancer a chaque fois !!!)

# In[ ]:


# # Diviser chaque document en morceaux plus petits
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)

# # Générer et persister les embeddings
# persist_directory = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_M-A\\Data\\ChromaDB"
# vectorstore = Chroma.from_documents(
#     documents=splits,
#     embedding=OpenAIEmbeddings(),
#     persist_directory=persist_directory
# )

# vectorstore.persist()
# print("Embeddings générés et stockés dans ChromaDB.")

# ### Génération avec LLM

# ### RAG MULTI QUERY

# In[ ]:


def rag_query(question):
        # Placez tout votre code ici
        # Multi Query: Different Perspectives
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = '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'

    os.environ['OPENAI_API_KEY'] = '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'

    # Recharger le vectorstore depuis le répertoire persist_directory
    vectorstore = Chroma(
        persist_directory="C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_M-A\\Data\\ChromaDB",
        embedding_function=OpenAIEmbeddings()
    )

    print("VectorStore chargé depuis le disque.")


    # Créer un système de récupération
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Utiliser Maximal Marginal Relevance
        search_kwargs={
            "k": 10,  # Récupérer plus de documents
            "score_threshold": 0.01  # Réduire le seuil de score pour inclure plus de résultats
        }
    )

    template = """You are an AI language model assistant. Your task is to generate ten 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)


    generate_queries = (
        prompt_perspectives 
        | ChatOpenAI(model= 'o1-mini') 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    def get_unique_union(documents: list[list]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]
    
    # Retrieve
    question = "Quelles nouveautées pour Vulcain ingénierie ?"
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":question})

    # RAG
    template = """Tu es un assistant chatbot qui travaille dans un cabinet de finance d'entreprise. Ton rôle est de donner les informations les plus pertinentes possibles en te basant sur les sources que tu as. Voici le contexte pour t'aider:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model='o1-mini')

    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    final_rag_chain.invoke({"question":question})
    answer = final_rag_chain.invoke({"question": question, "context": docs})
    return answer


# ### RAG FUSION

# In[ ]:

import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from json import dumps, loads
from langchain.schema import Document
from langchain.vectorstores import FAISS

def rag_fusion(question: str) -> str:
    # Configurer les clés API
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = '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'

    os.environ['OPENAI_API_KEY'] = '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'

    faiss_index_path = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_MnA\\Data\\FAISS_index"
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
            faiss_index_path, 
            embeddings=embedding, 
            allow_dangerous_deserialization=True
        )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "score_threshold": 0.01
        }
    )

    # Prompt pour la génération des requêtes (base complète)
    query_generation_template = """You are a seasoned M&A consultant with access to a broad dataset that includes recent news, company profiles, and investment fund details. Given the user's question:

    {question}

    Generate exactly 4 focused queries that will help retrieve the most relevant and comprehensive information from this rich dataset. The queries should cover aspects of M&A news, company insights, and fund activities if relevant. Aim to find highly relevant and up-to-date details that answer the user's inquiry.

    Output 4 queries:
    """
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)

    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(model='o1-mini')
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    queries = generate_queries.invoke({"question": question})
    print(f"Requêtes générées : {queries}")

    results = [retriever.invoke(q) for q in queries]
    print(f"Documents récupérés : {results}")

    # Fusion RRF
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            doc_str = dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)

    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [loads(d_str)]
    ]

    print(f"Documents fusionnés : {len(reranked_docs)} documents rerankés.")

    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])

    # Prompt pour la réponse finale (base complète)
    answer_template = """You are an M&A expert who can analyze recent news, company data, and fund information to provide a comprehensive and accurate answer. Using the following context extracted from various M&A-related sources (news, companies, funds), answer the user's question concisely and factually. Highlight relevant deals, company details, or fund strategies if mentioned. Do not invent information that isn't provided. Always give your source.It's in the name of the document you are using the information (it's Arx or CFNews. Do no say that it comes from the context, always say that it is from Arx or CFNews from the name of the doc in the metadata)).

    Context:
    {context}

    Question: {question}

    Provide a clear, fact-based answer focusing on the M&A domain.
    """
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')

    final_input = {"context": context, "question": question}
    answer = (
        answer_prompt
        | llm
        | StrOutputParser()
    ).invoke(final_input)

    return answer

def rag_fusion_actualites(question: str) -> str:
    # Configurer les clés API
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = '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'

    os.environ['OPENAI_API_KEY'] = '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'

    faiss_index_path = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_MnA\\Data\\FAISS_index_actualites"
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
            faiss_index_path, 
            embeddings=embedding, 
            allow_dangerous_deserialization=True
        )

    # Créer un système de récupération
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Utiliser Maximal Marginal Relevance
        search_kwargs={
            "k": 10,
            "score_threshold": 0.01
        }
    )

    # Prompt pour la génération des requêtes
    query_generation_template = """You are a knowledgeable M&A news analyst. Your role is to generate multiple targeted search queries to retrieve the most relevant and recent M&A news from a specialized news database.

    Given the user's question: {question}

    Generate exactly 4 specific and focused queries related to recent M&A news, announcements, deals, or trends. Ensure these queries help in finding up-to-date and fact-based information about the topic mentioned by the user.
    Output 4 queries:
    """

    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)

    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(model='o1-mini')
        | StrOutputParser()
        | (lambda x: x.split("\n"))  # Liste des requêtes
    )

    # Étape 1 : Générer les requêtes
    queries = generate_queries.invoke({"question": question})
    print(f"Requêtes générées : {queries}")

    # Étape 2 : Récupération des documents
    results = [retriever.invoke(q) for q in queries]
    print(f"Documents récupérés : {results}")

    # Étape 3 : Fusion Reciprocal Rank Fusion
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            doc_str = dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)

    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [loads(d_str)]
    ]

    print(f"Documents fusionnés : {len(reranked_docs)} documents rerankés.")

    # Préparer le contexte pour le modèle LLM
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])

    # Prompt pour la réponse finale
    answer_template = """You are a financial journalist and M&A expert focusing on recent news. Using the following context extracted from M&A news sources, answer the user's question factually and succinctly. Highlight relevant and recent deals, events, or trends mentioned in the context. Do not invent information not provided in the context. Always give your source. It's in the name of the document you are using the information (it's Arx or CFNews. Do no say that it comes from the context, always say that it is from Arx or CFNews from the name of the doc in the metadata).

    Context:
    {context}

    Question: {question}
    
    If you are asked to do a market sheet or a company profile, structure the answer you give me based on the context.
    Else, provide a clear and fact-based answer drawn from the provided context.
    """
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')

    # Étape 4 : Génération de la réponse
    final_input = {"context": context, "question": question}
    answer = (
        answer_prompt
        | llm
        | StrOutputParser()
    ).invoke(final_input)

    return answer

def rag_fusion_fonds(question: str) -> str:
    # Configurer les clés API
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = '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'

    os.environ['OPENAI_API_KEY'] = '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'

    faiss_index_path = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_MnA\\Data\\FAISS_index_fonds"
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
            faiss_index_path, 
            embeddings=embedding, 
            allow_dangerous_deserialization=True
        )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.01
        }
    )

    # Prompt pour la génération des requêtes (fonds)
    query_generation_template = """You are an expert in private equity and investment funds. The user has asked a question related to investment funds, their strategies, sectors, geographic focus, or recent deals. Given the user's query:

    {question}

    Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
        """
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)

    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(model='o1-mini')
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    queries = generate_queries.invoke({"question": question})
    print(f"Requêtes générées : {queries}")

    results = [retriever.invoke(q) for q in queries]
    print(f"Documents récupérés : {results}")

    # Fusion RRF
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            doc_str = dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)

    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [loads(d_str)]
    ]

    print(f"Documents fusionnés : {len(reranked_docs)} documents rerankés.")

    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])

    # Prompt pour la réponse finale (fonds)
    answer_template = """You are a private equity and investment fund specialist. Using the following context sourced from a database of investment funds, answer the user's question with a focus on fund characteristics such as ticket size, sector preferences, geographic focus, and investment strategies. Provide a factual, clear, and concise explanation based solely on the provided information. Do not invent details not found in the context. Always give your source.It's in the name of the document you are using the information (it's Arx or CFNews. Do no say that it comes from the context, always say that it is from Arx or CFNews from the name of the doc in the metadata)).

    Context:
    {context}

    Question: {question}

    Offer a fact-based, to-the-point response, highlighting key investment criteria or fund details.
    """
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')

    final_input = {"context": context, "question": question}
    answer = (
        answer_prompt
        | llm
        | StrOutputParser()
    ).invoke(final_input)

    return answer

def rag_fusion_fiche_societe_to_word(question: str) -> dict:
    """
    Interroge la base d'actualités M&A via RAG et retourne une réponse structurée
    adaptée pour remplir un template Word.
    
    :param question: La question posée par l'utilisateur.
    :return: Un dictionnaire contenant les informations structurées pour le template Word.
    """
    # Configurer les clés API (à sécuriser, par exemple en utilisant des variables d'environnement)
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = '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'
    os.environ['OPENAI_API_KEY'] = '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'

    
    # Chemin vers l'index FAISS des actualités
    faiss_index_path = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_MnA\\Data\\FAISS_index_actualites"
    
    # Initialiser les embeddings et charger le vectorstore FAISS
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        faiss_index_path, 
        embeddings=embedding, 
        allow_dangerous_deserialization=True
    )

    # Créer un système de récupération avec Maximal Marginal Relevance (MMR)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,  # Nombre de documents à récupérer
            "score_threshold": 0.01  # Seuil de score pour inclure des résultats
        }
    )

    # Prompt pour la génération des requêtes
    query_generation_template = """You are a knowledgeable M&A news analyst. Your role is to generate multiple targeted search queries to retrieve the most relevant and recent M&A news from a specialized news database. Always give your source.It's in the name of the document you are using the information (it's Arx or CFNews. Do no say that it comes from the context, always say that it is from Arx or CFNews from the name of the doc in the metadata)).

    Given the user's question: {question}

    Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    """
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)

    # Générer les requêtes
    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(model='o1-mini')
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    # Étape 1 : Générer les requêtes
    queries = generate_queries.invoke({"question": question})
    print(f"Requêtes générées : {queries}")

    # Étape 2 : Récupération des documents
    results = [retriever.invoke(q) for q in queries]
    print(f"Documents récupérés : {results}")

    # Étape 3 : Fusion Reciprocal Rank Fusion (RRF)
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            doc_str = json.dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)  # Pondération RRF

    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [json.loads(d_str)]
    ]

    print(f"Documents fusionnés : {len(reranked_docs)} documents rerankés.")

    # Préparer le contexte pour le modèle LLM
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])

    # Prompt pour la réponse finale structuré en JSON
    answer_template = """You are a financial journalist and M&A expert. You MUST answer in JSON format only, strictly matching the provided structure. Always give your source.It's in the name of the document you are using the information (it's Arx or CFNews. Do no say that it comes from the context, always say that it is from Arx or CFNews from the name of the doc in the metadata))..

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
        "actualites_presse": "Give as much information as possilble (that you have)",
        "equity_story": "Provide details of investments or equity-related events. Give as much détails as you can. Present and order it for your client !",
        "creation": "Provide creation details or year of founding",
        "acquisition": "Give as much information as possilble (that you have)"
    }}

    DO NOT include any additional text before or after."""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')  # Utilisez le modèle approprié

    # Étape 4 : Génération de la réponse structurée
    final_input = {"context": context, "question": question}
    answer = (
        answer_prompt
        | llm
        | StrOutputParser()
    ).invoke(final_input)

    try:
        print("Réponse du LLM:", answer)
        # Convertir la réponse JSON en dictionnaire
        answer_dict = json.loads(answer)
    except json.JSONDecodeError:
        # Gérer les erreurs de parsing JSON
        answer_dict = {}
        print("Erreur lors du parsing JSON de la réponse du LLM.")

    return answer_dict

from docxtpl import DocxTemplate
import datetime
import os

def generate_fiche_societe(llm_response: dict, template_path: str, output_path: str):
    """
    Remplit un template Word avec les données de la réponse du LLM.

    :param llm_response: dict contenant les informations de la fiche société
    :param template_path: chemin vers le template Word
    :param output_path: chemin où le document rempli sera sauvegardé
    """
    doc = DocxTemplate(template_path)
    
    # Ajouter la date actuelle si elle n'est pas fournie
    if 'date' not in llm_response or not llm_response['date']:
        llm_response['date'] = datetime.datetime.now().strftime("%Y-%m-%d")
    
    doc.render(llm_response)
    doc.save(output_path)

def rag_fusion_multiples_transactions_comparables(question: str) -> str: 
    # Configurer les clés API
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = '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'
    os.environ['OPENAI_API_KEY'] = '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'

    faiss_index_path = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_MnA\\Data\\FAISS_index_multiples"
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
            faiss_index_path, 
            embeddings=embedding, 
            allow_dangerous_deserialization=True
        )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.01
        }
    )

    # Prompt pour la génération des requêtes (transactions comparables)
    query_generation_template = """You are an expert in mergers, acquisitions, and financial transactions. 
    The user has asked a question related to comparable transactions or multiples such as EV/Revenue, EV/EBITDA, or EV/EBIT. 
    Given the user's query:

    {question}

    Your task is to generate five alternative versions of the user question to retrieve relevant documents 
    from a vector database. The goal is to capture different perspectives of the user's query to ensure relevant 
    documents are retrieved. Provide these alternative questions separated by newlines.
    """
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)

    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(model='o1-mini')
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    queries = generate_queries.invoke({"question": question})
    print(f"Requêtes générées : {queries}")

    results = [retriever.invoke(q) for q in queries]
    print(f"Documents récupérés : {results}")

    # Fusion RRF
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            doc_str = dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)

    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [loads(d_str)]
    ]

    print(f"Documents fusionnés : {len(reranked_docs)} documents rerankés.")

    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])

    # Prompt pour la réponse finale (transactions comparables)
    answer_template = """You are an expert in financial transactions and valuation multiples. Using the following 
    context sourced from a database of comparable transactions, provide insights about relevant transactions, 
    key valuation multiples (e.g., EV/Revenue, EV/EBITDA), and deal characteristics.

    Context:
    {context}

    Question: {question}

    Your response should be factual, concise, and focused solely on the provided context. Include specific 
    multiples, transaction details, and other financial metrics when possible. Always indicate the source of 
    your information (it will always be MergerMarket.)
    """
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')

    final_input = {"context": context, "question": question}
    answer = (
        answer_prompt
        | llm
        | StrOutputParser()
    ).invoke(final_input)

    return answer
