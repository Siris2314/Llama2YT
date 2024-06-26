# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
device = "cuda"
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import json
import streamlit as st

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores import Chroma, FAISS
import nest_asyncio

from langchain.schema.runnable import RunnablePassthrough

from distill import load_speech_recognition_model, download_audio_from_youtube

import textwrap

from playwright.async_api import async_playwright

# from transformers import pipeline

import asyncio

import os

import torch

import requests

from langchain_together import ChatTogether

import sys

from dotenv import load_dotenv


load_dotenv('.env')

def get_webpage_size(url):
    response = requests.get(url)
    return len(response.content)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_prompt(instruction, sys_prompt):
    system_prompt = B_SYS + sys_prompt + E_SYS
    template = B_INST + system_prompt +  instruction + E_INST
    return template



filename = "yes.mp3"  # Replace with the actual filename

if os.path.exists(filename):
    os.remove(filename)
    print(f"File '{filename}' deleted successfully.")

def load_transcript(url):

    whisper = load_speech_recognition_model()

    audio_file = download_audio_from_youtube(
                url)

    transcription = whisper(
                audio_file, chunk_length_s=30, stride_length_s=5, batch_size=8
            )

    with open("new_transcript.txt", "w") as f:
                f.write(transcription["text"])

    print("Transcription saved to new_transcript.txt")


print(os.getenv("env"))


def load_tokenizer_and_llm():

    llm = ChatTogether(
        model = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
        max_tokens = 2048,
        together_api_key = os.getenv("TOGETHER_API_KEY")
    )

    return llm


instruction = "Given the context that has been provided. \n {context}, Answer the following question: \n{question}"

sys_prompt = """You are an expert in YouTube video question and answering.
You will be given context to answer from. Answer the questions with as much detail as possible and only in paragraphs.
In case you do not know the answer, you can say "I don't know" or "I don't understand".
In all other cases provide an answer to the best of your ability."""

prompt_sys = get_prompt(instruction, sys_prompt)


template = PromptTemplate(template=prompt_sys, input_variables=['context', 'question'])

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    
    wrapped_text = '\n'.join(wrapped_lines)
    
    return wrapped_text

def process_llm_response(llm_response):
    response_text = wrap_text_preserve_newlines(llm_response['text'])
    
    # Extracting sources into a list
    sources_list = [source.metadata['source'] for source in llm_response['context']]

    # Returning a dictionary with separate keys for text and sources
    return {"answer": response_text, "sources": sources_list}

def data_loader(url):
    load_transcript(url)
    loader =  TextLoader(f'new_transcript.txt')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5",
                                         model_kwargs={'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}, encode_kwargs={'normalize_embeddings': True})
    

    db = FAISS.from_documents(texts, 
                               embeddings)
    
    return db


def process_query(query, llm, db):
    retriever = db.as_retriever()
    llm_chain = LLMChain(llm=llm, prompt=template)
    rag_chain = ( 
        {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )
    ans = rag_chain.invoke(query)
    return process_llm_response(ans)


