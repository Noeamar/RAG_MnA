#!/usr/bin/env python
# coding: utf-8

# In[28]:


import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
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


os.environ['OPENAI_API_KEY'] = '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'

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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.load import dumps, loads

def rag_fusion(question):
    # Charger les clés API depuis les variables d'environnement
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    langchain_api_key = os.environ.get('LANGCHAIN_API_KEY')
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    
    if not langchain_api_key or not openai_api_key:
        raise ValueError("Les clés API OpenAI et LangChain ne sont pas configurées.")
    
    # Recharger le vectorstore depuis le répertoire persist_directory
    persist_directory = "C:\\Users\\namar\\Documents\\poc_RAG\\Projet_test\\RAG_M-A\\Data\\ChromaDB"
    vectorstore = Chroma(
        persist_directory=persist_directory,
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

    # Prompt pour générer les requêtes
    query_generation_template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)

    generate_queries = (
        prompt_rag_fusion 
        | ChatOpenAI(model='o1-mini')
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    def reciprocal_rank_fusion(results: list[list], k=60):
        """ Reciprocal_rank_fusion """
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        return [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

    # Génération des requêtes et récupération des documents
    queries = generate_queries.invoke({"question": question})
    results = [retriever.invoke(q) for q in queries]
    reranked_docs = reciprocal_rank_fusion(results)

    # Préparer les documents pour le modèle LLM
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])

    # Prompt pour répondre à la question
    answer_template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model='o1-mini')

    # Générer la réponse finale
    final_rag_chain = (
        {"context": context, "question": question} 
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    answer = final_rag_chain.invoke({"question": question, "context": context})

    return answer
