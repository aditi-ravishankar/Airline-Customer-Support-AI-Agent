import os
from pathlib import Path
from backend_utility import load_vectorstore_from_disk, create_and_save_vectorstore, is_small_talk_or_greeting, is_farewell, is_filler, get_sentiment_and_intent, generate_assistant_response, load_phrases_from_file, classify_intent, classify_sub_intent
from frontend_utility import add_bg_from_local, display_message, display_typing_effect
from action_flow import book_ticket_action, cancel_ticket_action
import time

# ----------------------------
# ENVIRONMENT & CACHE SETUP
# ----------------------------

# 1. Set ALL possible cache locations (new HuggingFace versions need this)
cache_dir = Path("D:/huggingface_cache")
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
import nest_asyncio
import random

# Hugging Face and Transformers
from transformers import logging

# LangChain and other libraries
from langchain.memory import ConversationSummaryBufferMemory

# LangChain Community
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain

# Streamlit
import streamlit as st

# ----------------------------
# STREAMLIT UI CONFIGURATION
# ----------------------------

st.set_page_config(page_title="SkyWings Airline Assistant ✈️", layout="wide")

# Styling to ensure the container and content stretch across the screen
st.markdown("""
     <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500&display=swap');
        
    /* Make sure the entire container takes full width and is centered */
    .container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        text-align: center;
        width: 100%;
        max-width: 100%;  /* Ensure no limit is placed on container width */
        margin: 0;  /* Remove any margin that may be restricting the width */
    }
        
    /* Title and text styling */
    .title {
        font-size: 48px !important;
        color: #000080;
        font-family: 'Quicksand', sans-serif;
        margin-bottom: 10px !important;
        }
    .header {
        font-size: 30px !important;
        color: #DC143C;
        font-family: 'Quicksand', sans-serif;
        margin-bottom: 8px !important;
    }
    .body {
        font-size: 18px !important;
        color: #000;
        font-family: 'Quicksand', sans-serif;
        margin-top: 5px !important;
    }

    /* Stretch the content within the page */
    .stApp {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Ensure that sidebar and main body respect the wide layout */
    .css-1d391kg {
        max-width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Add the container to your Streamlit app
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<p class="title">SkyWings Airline Assistant ✈️</p>', unsafe_allow_html=True)
st.markdown('<p class="header">Welcome to SkyWings</p>', unsafe_allow_html=True)
st.markdown('<p class="body">How can we assist you today?</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# Fix Streamlit event loop
nest_asyncio.apply()
logging.set_verbosity_error()  # Reduce warnings

# ----------------------------
# Initialise LLM
# ----------------------------

llm = Ollama(model="mistral", temperature = 0.8)  # Updated Ollama class

# ----------------------------
# Embedding & Retrieval
# ----------------------------

# Load or create vector store
vectordb = load_vectorstore_from_disk()

if not vectordb:
    directory_path_policies = "/Users/aditiravishankar/Desktop/Capstone/28 APRIL iisc_capstone-main/SkyWings Policy Document"
    # Fallback: Create a new index from sample documents if no vectorstore is found
    vectordb = create_and_save_vectorstore(directory_path_policies)
    
retriever = vectordb.as_retriever()
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# ----------------------------
# Sidebar: Controls
# ----------------------------
st.sidebar.button("🪠 Clear Chat", on_click=lambda: st.session_state.update(
    messages=[],
    display_stage=0,
    current_message=None
))

st.sidebar.markdown("---")
st.sidebar.markdown("📞 **Customer Support**")
st.sidebar.markdown("**Phone:** +91 (0)124 435 2500")
st.sidebar.markdown("**Email:** support@skywings.com")

st.sidebar.markdown("---")
st.sidebar.markdown("🕐 **Support Hours**")
st.sidebar.markdown("Mon–Sat: 9:00 AM – 8:00 PM")

st.sidebar.markdown("---")
st.sidebar.markdown("📍 **Head Office**")
st.sidebar.markdown("SkyWings HQ, MG Road, Bangalore")

st.sidebar.markdown("---")
st.sidebar.markdown("🌐 [Visit our website](https://skywings.com)")

# ----------------------------
# Session Initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_stage" not in st.session_state:
    st.session_state.display_stage = 0
if "current_message" not in st.session_state:
    st.session_state.current_message = None
if "form_mode" not in st.session_state:
    st.session_state.form_mode = None

# ----------------------------
# Session Initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_stage" not in st.session_state:
    st.session_state.display_stage = 0
if "form_mode" not in st.session_state:
    st.session_state.form_mode = None
if "current_message" not in st.session_state:
    st.session_state.current_message = None

# ----------------------------
# Main Chat Interface
# ----------------------------
add_bg_from_local("/Users/aditiravishankar/Desktop/Customer Support AI Agent/data/Images/Background1.jpg")  # your background

# Canned responses
greetings_responses = load_phrases_from_file("/Users/aditiravishankar/Desktop/Customer Support AI Agent/data/assistant_responses/greetings_responses.txt")
farewell_responses = load_phrases_from_file("/Users/aditiravishankar/Desktop/Customer Support AI Agent/data/user_greetings_farewells/farewell_phrases.txt")
filler_responses = load_phrases_from_file("/Users/aditiravishankar/Desktop/Customer Support AI Agent/data/user_greetings_farewells/filler_phrases.txt")

# Display history
for msg in st.session_state.messages:
    display_message(msg, show_analysis=(msg["role"] == "Customer"))

# Paths for your intent model
intent_model_path = "/Users/aditiravishankar/Desktop/Capstone/Best Bert Model/Best_Bert_Model"
intent_label_encoder_path = "/Users/aditiravishankar/Desktop/Capstone/Best Bert Model/Best_Bert_Model/label_encoder.pkl"

# 2. Chat input handling (Stage 0)
user_input = st.chat_input("Say something…")
if user_input and st.session_state.display_stage == 0:
    # 1) Sentiment & intent
    sentiment_result, intent_label = asyncio.run(
        get_sentiment_and_intent(user_input, intent_model_path, intent_label_encoder_path)
    )
    sentiment_label = sentiment_result[0]["label"]
    sentiment_emoji = {"POSITIVE":"😄","NEGATIVE":"😞","NEUTRAL":"😐"}[sentiment_label.upper()]
    assistant_response = ""

    # 2) Canned replies
    if is_small_talk_or_greeting(user_input):
        assistant_response = random.choice(greetings_responses)
        intent_label = "Greeting / Small Talk"
    elif is_farewell(user_input):
        assistant_response = random.choice(farewell_responses)
        intent_label = "Farewell"
    elif is_filler(user_input):
        assistant_response = random.choice(filler_responses)
        intent_label = "Filler Words"
    else:
        # 3) Booking / cancellation detection
        intent_label = classify_intent(user_input, intent_model_path, intent_label_encoder_path)
        if intent_label == "Booking, Modifications And Cancellations":
            sub_intent, score = classify_sub_intent(user_input)
            if sub_intent == "Booking Request" and score >= 0.5:
                # show booking form page
                st.session_state.form_mode = "booking"
                st.session_state.current_message = {
                    "role": "Customer",
                    "content": user_input,
                    "sentiment": f"{sentiment_label} {sentiment_emoji}",
                    "intent": intent_label,
                    "assistant_logic": assistant_response,
                    "response": None
                }
                st.session_state.display_stage = 4
                st.rerun()  # Rerun the app to process the form (avoid appending assistant response yet)
            elif sub_intent == "Cancellation Request" and score >= 0.5:
                # show cancellation form page
                st.session_state.form_mode = "cancellation"
                st.session_state.current_message = {
                    "role": "Customer",
                    "content": user_input,
                    "sentiment": f"{sentiment_label} {sentiment_emoji}",
                    "intent": intent_label,
                    "assistant_logic": assistant_response,
                    "response": None
                }
                st.session_state.display_stage = 4
                st.rerun()  # Rerun the app to process the form (avoid appending assistant response yet)
            else:
                assistant_response = generate_assistant_response(intent_label, user_input)
        else:
            assistant_response = generate_assistant_response(intent_label, user_input)

    # 4) If not in form flow, append both messages
    if st.session_state.display_stage == 0 and st.session_state.form_mode is None:
        # Store assistant response in current_message with calculated sentiment and intent
        st.session_state.current_message = {
            "role": "Customer",
            "content": user_input,
            "sentiment": f"{sentiment_label} {sentiment_emoji}",
            "intent": intent_label,
            "assistant_logic": assistant_response,
            "response": None
        }

    st.session_state.display_stage = 1
    st.rerun()

elif st.session_state.display_stage == 1:
    # Add the user message only after sentiment and intent are calculated
    display_message(st.session_state.current_message, show_analysis=True)

    # Show typing animation (don't overwrite the previous chat bubble)
    response_placeholder = st.empty()
    response_placeholder.markdown(
        """
        <style>
        @keyframes blinkDots {
            0% { content: ""; }
            33% { content: "."; }
            66% { content: ".."; }
            100% { content: "..."; }
        }
        .typing-dots::after {
            content: "...";
            animation: blinkDots 1s steps(3, end) infinite;
        }
        </style>
        <div style='display: flex; justify-content: flex-end;'>
            <div style='background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 60%; text-align: right;'>
                <strong>✈️ SkyWings Assistant</strong><br>
                <span class='typing-dots'>Typing</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Generate assistant response only once
    if st.session_state.current_message["response"] is None:
        user_query = st.session_state.current_message["content"]
        assistant_logic = st.session_state.current_message["assistant_logic"]

        if assistant_logic == "vector_db":
            result = qa_chain.run(user_query)
        else:
            result = assistant_logic
        st.session_state.current_message["response"] = result

    st.session_state.display_stage = 2
    st.rerun()

elif st.session_state.display_stage == 2:
    # Show enriched user message again (with sentiment/intent)
    display_message(st.session_state.current_message, show_analysis=True)

    # Add typing animation placeholder for assistant's response
    response_placeholder = st.empty()
    assistant_response = st.session_state.current_message["response"]

    # Start typing effect
    if assistant_response:
        display_typing_effect(response_placeholder, assistant_response)

    # Store both messages after typing effect
    enriched_user_message = {
        "role": "Customer",
        "content": st.session_state.current_message["content"],
        "sentiment": st.session_state.current_message["sentiment"],
        "intent": st.session_state.current_message["intent"],
        "response": None
    }
    st.session_state.messages.append(enriched_user_message)
    st.session_state.messages.append({
        "role": "Assistant",
        "content": assistant_response.strip()
    })

    st.session_state.display_stage = 0
    st.session_state.current_message = None

# Form handling
elif st.session_state.display_stage == 3:
    display_message(st.session_state.current_message, show_analysis=True)
    if st.session_state.form_mode == "booking":
        with st.form("flight_booking_form"):
            st.subheader("🛫 Book a Flight")
            departure   = st.text_input("Departure City")
            arrival     = st.text_input("Arrival City")
            travel_date = st.date_input("Travel Date")
            submitted   = st.form_submit_button("Book Flight")
        if submitted:
            confirmation = book_ticket_action(departure, arrival, travel_date)
            confirmation = confirmation.replace('\n', '<br>')
            st.session_state.current_message["response"] = confirmation
            st.session_state.form_mode = None
            st.session_state.display_stage = 5
            st.rerun()

    elif st.session_state.form_mode == "cancellation":
        with st.form("cancellation_form"):
            st.subheader("❌ Cancel a Booking")
            pnr       = st.text_input("Enter your PNR")
            submitted = st.form_submit_button("Cancel Ticket")
        if submitted:
            confirmation = cancel_ticket_action(pnr)
            confirmation = confirmation.replace('\n', '<br>')
            st.session_state.current_message["response"] = confirmation
            st.session_state.form_mode = None
            st.session_state.display_stage = 5
            st.rerun()


elif st.session_state.display_stage == 4:
    # Add the user message only after sentiment and intent are calculated
    display_message(st.session_state.current_message, show_analysis=True)

    # Delay to allow user to see the bubble
    time.sleep(2)

    st.session_state.display_stage = 3
    st.rerun()

elif st.session_state.display_stage == 5:
    # Show user message again
    display_message(st.session_state.current_message, show_analysis=True)

    # Show assistant confirmation with typing
    response_placeholder = st.empty()
    assistant_response = st.session_state.current_message["response"]

    if assistant_response:
        display_typing_effect(response_placeholder, assistant_response)

    # Append both messages
    enriched_user_message = {
        "role": "Customer",
        "content": st.session_state.current_message["content"],
        "sentiment": st.session_state.current_message["sentiment"],
        "intent": st.session_state.current_message["intent"],
        "response": None
    }
    st.session_state.messages.append(enriched_user_message)
    st.session_state.messages.append({
        "role": "Assistant",
        "content": assistant_response.strip()
    })

    # Reset for next round
    st.session_state.display_stage = 0
    st.session_state.current_message = None




