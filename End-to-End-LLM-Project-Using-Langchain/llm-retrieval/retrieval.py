#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


# In[2]:


#load openAI api key
os.environ['OPENAI_API_KEY'] = 'openai_key_add'


# In[3]:


# Initialise LLM with required params
llm = OpenAI(temperature=0.9, max_tokens=500)


# ### (1) Load data

# In[4]:


loaders = UnstructuredURLLoader(urls=[
    "https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html"
])
data = loaders.load() 
len(data)


# # (2) Split data to create chunks

# In[5]:


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
docs = text_splitter.split_documents(data)


# In[6]:


len(docs)


# In[7]:


docs[0]


# # (3) Create embeddings for these chunks and save them to FAISS index

# In[8]:


# Create the embeddings of the chunks using openAIEmbeddings
embeddings = OpenAIEmbeddings()

# Pass the documents and embeddings inorder to create FAISS vector index
vectorindex_openai = FAISS.from_documents(docs, embeddings)


# In[ ]:


# # Storing vector index create in local
# file_path="vector_index.pkl"
# with open(file_path, "wb") as f:
#     pickle.dump(vectorindex_openai, f)


# In[11]:


vectorindex_openai = FAISS.from_documents(docs, embeddings)

vectorindex_openai.save_local("faiss_store")


# In[18]:


if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorIndex = FAISS.load_local("faiss_store", OpenAIEmbeddings())


# ## (4) Retrieve similar embeddings for a given question and call LLM to retrieve final answer

# In[19]:


chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
chain


# In[21]:


query = "what is the price of Tiago iCNG?"
# query = "what are the main features of punch iCNG?"

langchain.debug=True

chain({"question": query}, return_only_outputs=True)


# In[ ]:




