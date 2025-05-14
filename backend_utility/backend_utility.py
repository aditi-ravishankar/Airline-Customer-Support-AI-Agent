# Standard libraries
import asyncio
import nest_asyncio
import pickle
import torch
import string
import base64
import random
import time

import os
from pathlib import Path

# Hugging Face and Transformers
from transformers import logging
from transformers import BertForSequenceClassification, BertTokenizer

# LangChain and other libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationSummaryBufferMemory

# LangChain Community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain

# Pipeline
from transformers import pipeline

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Save the FAISS vectorstore to disk
def save_vectorstore(vectordb):
    vectordb.save_local("faiss_vectorstore")

# Load FAISS vectorstore from disk
def load_vectorstore_from_disk():
    if os.path.exists("faiss_vectorstore"):
        embedding_model = embeddings  # Use the same embedding model
        return FAISS.load_local(
            "faiss_vectorstore",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        return None

# Create and save a new vectorstore from documents in a directory
def create_and_save_vectorstore(directory_path):
    docs = []
    loader = TextLoader('')
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for filepath in Path(directory_path).rglob('*.txt'):
        loader.file_path = str(filepath)
        document = loader.load()
        split_docs = splitter.split_documents(document)
        for doc in split_docs:
            doc.metadata["source"] = filepath.name
        docs.extend(split_docs)

    vectordb = FAISS.from_documents(docs, embeddings)
    save_vectorstore(vectordb)
    return vectordb

# Load model with neutral detection (3-class)
sentiment_pipe = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=-1
)

# BERT Intent Classification function
def classify_intent(query, model_path, label_encoder_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Force the use of CPU
    device = torch.device("cpu")  # Forces use of CPU
    model.to(device)

    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)

    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        intent = le.inverse_transform([predicted_class])[0]

    return intent

# Function to run sentiment and intent classification in parallel
async def get_sentiment_and_intent(user_input, intent_model_path, intent_label_encoder_path):
    sentiment_future = asyncio.to_thread(sentiment_pipe, user_input)
    intent_future = asyncio.to_thread(classify_intent, user_input, intent_model_path, intent_label_encoder_path)
    
    sentiment_result, intent_label = await asyncio.gather(sentiment_future, intent_future)
    return sentiment_result, intent_label

# Get the list of possible phrases from a pre-defined list
def load_phrases_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# Clean the user's phrase by removing punctuations and converting it to lowercase
def clean_phrase(user_input):
    # Remove punctuation from input and convert to lowercase
    return user_input.translate(str.maketrans("", "", string.punctuation)).strip().lower()

# Greeting detection function
def is_small_talk_or_greeting(user_input):
    small_talk_phrases = load_phrases_from_file('/Users/aditiravishankar/Desktop/Customer Support AI Agent/data/user_greetings_farewells/greetings_small_talk_phrases.txt')
    # Clean the user input
    user_input_clean = clean_phrase(user_input)
    
    # Clean the small talk phrases and check if any match
    small_talk_phrases_cleaned = [clean_phrase(phrase) for phrase in small_talk_phrases]
    
    # Check if any small talk or greeting phrase matches the cleaned user input
    return user_input_clean in small_talk_phrases_cleaned

# Farewell detection function
def is_farewell(user_input):
    farewell_phrases = load_phrases_from_file('/Users/aditiravishankar/Desktop/Customer Support AI Agent/data/user_greetings_farewells/farewell_phrases.txt')
    
    # Clean the user input
    user_input_clean = clean_phrase(user_input)
    
    # Clean the farewell phrases and check if any match
    farewell_phrases_cleaned = [clean_phrase(phrase) for phrase in farewell_phrases]
    
    # Check if any farewell phrase matches the cleaned user input
    return user_input_clean in farewell_phrases_cleaned

# Filler word detection function
def is_filler(user_input):
    filler_phrases = load_phrases_from_file('/Users/aditiravishankar/Desktop/Customer Support AI Agent/data/user_greetings_farewells/filler_phrases.txt')
    
    # Clean the user input
    user_input_clean = clean_phrase(user_input)
    
    # Clean the farewell phrases and check if any match
    filler_phrases_cleaned = [clean_phrase(phrase) for phrase in filler_phrases]
    
    # Check if any farewell phrase matches the cleaned user input
    return user_input_clean in filler_phrases_cleaned

# Load zero-shot classifier
sub_intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Sub-intent labels
sub_intents = [
    "Booking Request",
    "Cancellation Request",
    "Modification Request",
    "Policy Question"
]

def classify_sub_intent(user_query):
    result = sub_intent_classifier(user_query, sub_intents, multi_label=False)
    return result['labels'][0], result['scores'][0]

# Assistant's polite response generator
def generate_assistant_response(intent, user_query):
    special_responses = {
        "Other": (
            "I'm not sure about that, but I can help with other queries! "
            "For assistance with this matter, you can reach us at Tel: +91 (0)124 4973838, "
            "or email customer-support@skywings.com. You can also reach out to us on Facebook (https://www.facebook.com/goSkyWings.in) "
            "or Twitter (@SkyWings6E) for feedback, concerns, or comments."
        ),
        "Irrelevant": "Sorry, that's out of my scope. Could you please refine your question?",
    }

    if intent in special_responses:
        return special_responses[intent]
    else:
        return "vector_db"  # Signal to use vector retrieval for actual answer