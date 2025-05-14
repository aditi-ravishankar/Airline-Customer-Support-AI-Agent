import os
from pathlib import Path
from backend_utility import load_vectorstore_from_disk, create_and_save_vectorstore, is_small_talk_or_greeting, is_farewell, is_filler, get_sentiment_and_intent, generate_assistant_response, load_phrases_from_file, classify_intent, classify_sub_intent

# ----------------------------
# ENVIRONMENT & CACHE SETUP
# ----------------------------

# 1. Set ALL possible cache locations (new HuggingFace versions need this)
cache_dir = Path("D2:/huggingface_cache")
cache_dir.mkdir(exist_ok=True, parents=True)

os.environ['HF_HOME'] = str(cache_dir)
os.environ['HF_DATASETS_CACHE'] = str(cache_dir / "datasets")
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)

# Configure paths
os.environ["HF_HOME"] = "D:/huggingface_cache"  # Set before loading pipeline
os.environ['OLLAMA_MODELS'] = 'D:/ollama_models'  # Stores Ollama models here


# 2. Patch the cache before any imports (critical!)
import huggingface_hub.constants
huggingface_hub.constants.HF_HUB_CACHE = str(cache_dir)

# Now proceed with necessary imports

# Standard libraries
import asyncio
import pickle
import torch
import string
import random

# Hugging Face and Transformers
from transformers import BertForSequenceClassification, BertTokenizer

# LangChain and other libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationSummaryBufferMemory

# LangChain Community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain

# Pipeline
from transformers import pipeline

# Streamlit
import streamlit as st

from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
import requests
import traceback


# === Config ===
load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER')

app = Flask(__name__)

llm = Ollama(model="mistral", temperature = 0.8)  # Updated Ollama class

# Load or create vector store
vectordb = load_vectorstore_from_disk()

if not vectordb:
    directory_path_policies = "/Users/aditiravishankar/Desktop/Capstone/28 APRIL iisc_capstone-main/SkyWings Policy Document"
    # Fallback: Create a new index from sample documents if no vectorstore is found
    vectordb = create_and_save_vectorstore(directory_path_policies)

# Canned responses
greetings_responses = load_phrases_from_file("/Users/aditiravishankar/Desktop/Customer Support AI Agent/data/assistant_responses/greetings_responses.txt")
farewell_responses = load_phrases_from_file("/Users/aditiravishankar/Desktop/Customer Support AI Agent/data/user_greetings_farewells/farewell_phrases.txt")
filler_responses = load_phrases_from_file("/Users/aditiravishankar/Desktop/Customer Support AI Agent/data/user_greetings_farewells/filler_phrases.txt")
    
retriever = vectordb.as_retriever()
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
assistant_response = ''

# Define paths for intent classification model and label encoder file
intent_model_path = "/Users/aditiravishankar/Desktop/Capstone/Best Bert Model/Best_Bert_Model"
intent_label_encoder_path = "/Users/aditiravishankar/Desktop/Capstone/Best Bert Model/Best_Bert_Model/label_encoder.pkl"

# === WhatsApp Webhook ===
@app.route('/webhook', methods=['POST'])
def webhook():
    incoming_data = request.form
    user_msg = incoming_data.get('Body', '')
    from_number = incoming_data.get('From', '')

    print(f"[Webhook] Message from {from_number}: {user_msg}")

    # Process sentiment and intent first before adding user message
    sentiment_result, intent_label = asyncio.run(get_sentiment_and_intent(user_msg, intent_model_path, intent_label_encoder_path))


    # Greetings logic block
    if is_small_talk_or_greeting(user_msg):
        assistant_response = random.choice(greetings_responses)
        intent_label = "Greeting / Small Talk"

    elif is_farewell(user_msg):
        assistant_response = random.choice(farewell_responses)
        intent_label = "Farewell"
        
    elif is_filler(user_msg):
        assistant_response = random.choice(filler_responses)
        intent_label = "Filler Words"

    else:
        intent_label = classify_intent(user_msg, intent_model_path, intent_label_encoder_path)
        print("Got a non-greeting intent")
        if intent_label == "Booking, Modifications And Cancellations":
            print("Got a Booking, Modifications And Cancellations intent")
            sub_intent, score = classify_sub_intent(user_msg)
            print("Got subintent = ", sub_intent)
            print("Got score = ", score)
            if sub_intent == "Booking Request" and score >= 0.5:
                print("inside booking request action")
                assistant_response = "[Trigger booking request]"

            elif sub_intent == "Cancellation Request" and score >= 0.5:
                print("inside cancellation request action")
                assistant_response = "[Trigger cancellation request]"
            else:
                print("inside modification / policy subintent")
                assistant_response = generate_assistant_response(intent_label, user_msg)
        else:
            print("normal RAG flow")
            assistant_response = generate_assistant_response(intent_label, user_msg)
    
    print("Assistant response: ", assistant_response)
    
    result = ""

    if assistant_response == "vector_db":
            result = qa_chain.run(user_msg)
            print("assistance response is vector db")
            print("result: ", result)
    
    else:
        result = assistant_response
        print("result: ", result)

    reply_text = "Hmm, something went wrong."

    if result:
        reply_text = result
        print("reply_text: ", reply_text)

    # Send reply back via Twilio
    try:
        twilio_response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json",
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            data={
                "From": TWILIO_WHATSAPP_NUMBER,
                "To": from_number,
                "Body": reply_text
            },
            timeout=10
        )
        print(f"[Twilio] Status: {twilio_response.status_code}")
    except Exception as e:
        print(f"[Error] Failed to send Twilio message: {e}")
        traceback.print_exc()

    return Response(str(MessagingResponse()), mimetype="application/xml")

# === Run Server ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

