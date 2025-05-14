# Standard libraries
import base64
import time

# Pipeline
from transformers import pipeline

# Streamlit
import streamlit as st

# Set background image function
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
# Function for simulating token-by-token response with typing effect
def stream_response_tokens(response):
    """Simulate real-time token generation."""
    for word in response.split():
        yield word + " "  # Return one word at a time
        time.sleep(0.1)  # Simulate a delay between tokens to create the typing effect

def display_typing_effect(placeholder, response_text):
    """Display assistant message with typing effect using a Streamlit placeholder."""
    full_response = ""
    for token in stream_response_tokens(response_text):
        full_response += token
        placeholder.markdown(
            f"""
            <style>
            @keyframes blink {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0; }}
                100% {{ opacity: 1; }}
            }}
            .blinking-cursor {{
                animation: blink 1s step-start infinite;
                display: inline;
            }}
            </style>
            <div style='display: flex; justify-content: flex-end;'>
                <div style='background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 60%; text-align: right;'>
                    <strong>✈️ SkyWings Assistant</strong><br>
                    <span>{full_response}<span class='blinking-cursor'>|</span></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Final display without blinking cursor
        placeholder.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end;'>
                <div style='background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 60%; text-align: right;'>
                    <strong>✈️ SkyWings Assistant</strong><br>
                    <span>{full_response}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
def display_message(msg, show_analysis=False):
    """Helper function to display messages with proper formatting"""
    avatars = {"Customer": "🙋", "Assistant": "✈️"}
    colors = {"Customer": "#DCF8C6", "Assistant": "#F1F0F0"}
    
    if msg['role'] == "Customer":
        st.markdown(
            f"""
            <div style='display: flex; gap: 8px; margin-bottom: 10px;'>
                <div style='background-color: {colors['Customer']}; padding: 10px; border-radius: 10px; max-width: 45%; text-align: left;'>
                    <strong>{avatars['Customer']} Customer</strong><br>
                    <span>{msg['content']}</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        
        if show_analysis and 'sentiment' in msg:
            st.markdown(
                f"""
                <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-size: 14px; max-width: 35%; min-width: 150px;'>
                    <strong>🧠 Sentiment:</strong> {msg.get('sentiment', '')}<br>
                    <strong>🎯 Intent:</strong> {msg.get('intent', '')} {f"({msg.get('score', '')}%)" if msg.get('score') else ''}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif msg['role'] == "Assistant":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                <div style='background-color: {colors['Assistant']}; padding: 10px; border-radius: 10px; max-width: 60%; text-align: right;'>
                    <strong>{avatars['Assistant']}✈️ SkyWings Assistant</strong><br>
                    <span>{msg['content']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )