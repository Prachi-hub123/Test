# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:05:02 2024

@author: rohit
"""
import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader


# pip install unstructured[pdf] --user
# Update path variable in evironment variables as per warning during installation of unstructured=pdf

def get_llm(path):
    #path = 'C://Users/rohit/Documents/2024/LLM/data/'
    loader = DirectoryLoader(path)
    documents = loader.load()
    
    # We need to split the text using Character Text Split such that it should not increse token size
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    
    texts = text_splitter.split_documents(documents)
 
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # pip install tiktoken
    # pip install faiss-cpu
    vector_db = FAISS.from_documents(texts, embeddings)
    
    llm = OpenAI(openai_api_key=openai_api_key)
    
    
    qa = ConversationalRetrievalChain.from_llm(llm=llm, 
                                     chain_type="stuff",\
                                     retriever=vector_db.as_retriever())
                          
    #question = "which data types are used in gemini?"
    #qa.invoke({'question':question, 'chat_history':[]})
    return qa
    
