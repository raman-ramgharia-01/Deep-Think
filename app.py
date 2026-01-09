import streamlit as st
import json
import time
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from ResearchSystem import self_research

# Set page config
st.set_page_config(
    page_title="Data Science Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu, header, footer {visibility: hidden;}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 20px;
    }
    
    .sidebar-title {
        font-size: 24px;
        font-weight: 700;
        color: #000;
        margin-bottom: 5px;
    }
    
    .sidebar-subtitle {
        font-size: 14px;
        color: #666;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .divider {
        height: 1px;
        background-color: #e0e0e0;
        margin: 25px 0;
    }
    
    .section-title {
        font-size: 12px;
        color: #888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 15px;
    }
    
    .chat-item {
        padding: 12px 15px;
        margin-bottom: 8px;
        border-radius: 8px;
        background: white;
        border: 1px solid #e0e0e0;
        cursor: pointer;
        font-size: 14px;
        color: #333;
        transition: all 0.2s;
    }
    
    .chat-item:hover {
        background: #e9ecef;
        border-color: #007bff;
    }
    
    .chat-item.active {
        background: #007bff;
        color: white;
        border-color: #007bff;
    }
    
    .new-chat-btn {
        background: #007bff;
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        width: 100%;
        margin-top: 10px;
        transition: background-color 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    .new-chat-btn:hover {
        background: #0056b3;
    }
    
    /* Main area styling */
    .main-header {
        padding: 30px 0 20px 0;
    }
    
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #000;
        margin-bottom: 10px;
    }
    
    .main-subtitle {
        font-size: 16px;
        color: #666;
    }
    
    /* Messages area */
    .message-bubble {
        padding: 15px 20px;
        border-radius: 20px;
        margin: 15px 0;
        max-width: 70%;
        font-size: 16px;
        line-height: 1.5;
    }
    
    .user-bubble {
        background: #007bff;
        color: white;
        border-radius: 20px 20px 0 20px;
        margin-left: auto;
    }
    
    .assistant-bubble {
        background: #f8f9fa;
        color: #333;
        border-radius: 20px 20px 20px 0;
        border-left: 4px solid #007bff;
    }
    
    /* Welcome message */
    .welcome-container {
        text-align: center;
        padding: 60px 20px;
    }
    
    .welcome-icon {
        font-size: 48px;
        margin-bottom: 20px;
    }
    
    .welcome-title {
        font-size: 28px;
        font-weight: 600;
        color: #333;
        margin-bottom: 15px;
    }
    
    .welcome-text {
        font-size: 16px;
        color: #666;
        line-height: 1.6;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Chat input */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 20px;
        border-top: 1px solid #e0e0e0;
        z-index: 100;
    }
    
    .chat-input-wrapper {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Scrollable area */
    .scrollable-area {
        margin-bottom: 100px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #888;
        font-size: 13px;
        position: fixed;
        bottom: 80px;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = "default"

if 'all_chats' not in st.session_state:
    st.session_state.all_chats = {
        "default": {
            "id": "default",
            "title": "Welcome Chat",
            "messages": [],
            "created": datetime.now().strftime("%H:%M")
        }
    }

# Initialize no-answer phrases
NO_ANSWER_PHRASES = [
    "don't have enough information",
    "no information provided",
    "cannot answer",
    "not enough context",
    "i don't know",
    "i don't have",
    "sorry, i cannot"
]

class RAGSystem:
    def __init__(self, api_key=None):
        # Store API key
        self.api_key = api_key
        
        # Initialize model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load data
        try:
            self.df = joblib.load('normalize_data.joblib')
        except Exception as e:
            raise Exception(f"Failed to load data: {e}")
    
    def analyze_with_groq(self, text_data):
        """Send text to Groq API and get response"""
        try:
            # Use the stored API key
            api_key = self.api_key or st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
            
            if not api_key:
                raise Exception("API key not found")
            
            client = Groq(api_key=api_key)

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions about Data Science based on provided context."
                    },
                    {
                        "role": "user",
                        "content": text_data,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=500
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"API Error: {e}")
    
    def get_response(self, user_query):
        """Main function to process query and return response"""
        if self.df is None or len(self.df) == 0:
            raise Exception("System not properly initialized")
        
        # Get query embedding
        user_embedding = self.model.encode(user_query)
        reshaped_user_embedding = user_embedding.reshape(1, -1)
        normalized_user_embedding = normalize(reshaped_user_embedding, norm='max')[0]
        
        # Calculate similarities
        similarities = cosine_similarity(
            normalized_user_embedding.reshape(1, -1), 
            np.stack(self.df['embedding'].values)
        )
        
        # Get top results
        top_results = 3
        top_chunks = similarities[0].argsort()[-top_results:][::-1]
        
        # Build retrieved context
        retrieved_context = ""
        for idx in top_chunks:
            retrieved_context += self.df.iloc[idx]['text'] + "\n\n"
        
        # Create RAG prompt
        rag_prompt = f"""Context:
{retrieved_context}

Based on the context above, answer this question: {user_query}

If the context doesn't contain relevant information, say "I don't have enough information in the provided context to answer this question."

Answer:"""
        
        # Get response from Groq
        response = self.analyze_with_groq(rag_prompt)
        
        return response

# Initialize RAG system
def initialize_rag_system(api_key=None):
    """Initialize RAG system"""
    try:
        rag_system = RAGSystem(api_key)
        st.session_state.system_initialized = True
        st.session_state.rag_system = rag_system
        return True
    except Exception as e:
        return False

# Auto-initialize system
if not st.session_state.system_initialized:
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if api_key and os.path.exists("normalize_data.joblib"):
        if initialize_rag_system(api_key):
            st.success("✅ System initialized successfully!")
        else:
            st.error("❌ Failed to initialize system")

# Create new chat function
def create_new_chat():
    new_id = f"chat_{len(st.session_state.all_chats)}"
    st.session_state.current_chat_id = new_id
    st.session_state.conversation_history = []
    st.session_state.all_chats[new_id] = {
        "id": new_id,
        "title": f"Chat {len(st.session_state.all_chats)}",
        "messages": [],
        "created": datetime.now().strftime("%H:%M")
    }

# Load chat function
def load_chat(chat_id):
    if chat_id in st.session_state.all_chats:
        st.session_state.current_chat_id = chat_id
        st.session_state.conversation_history = st.session_state.all_chats[chat_id]["messages"].copy()

# ============================================
# SIDEBAR - Left Column
# ============================================

with st.sidebar:
    # Logo and Title
    st.markdown("""
    <div class="sidebar-title">Data Science<br>Assistant</div>
    <div class="sidebar-subtitle">
        <span>Powered by RAG • Llama 3.3</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Recent Chats Section
    st.markdown('<div class="section-title">CHATS</div>', unsafe_allow_html=True)
    
    # Display recent chats
    for chat_id, chat_data in list(st.session_state.all_chats.items())[-5:]:  # Show last 5 chats
        is_active = chat_id == st.session_state.current_chat_id
        if st.button(
            f"{chat_data['title']}",
            key=f"btn_{chat_id}",
            help=f"Created at {chat_data['created']}",
            use_container_width=True
        ):
            load_chat(chat_id)
            st.rerun()
    
    # New Chat Button
    if st.button("＋ New Chat", key="new_chat_btn", use_container_width=True):
        create_new_chat()
        st.rerun()

# ============================================
# MAIN AREA - Right Column
# ============================================

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">How can I help you?</div>
    <div class="main-subtitle">Ask me anything about Data Science, Machine Learning, or related topics</div>
</div>
""", unsafe_allow_html=True)

# Scrollable messages area
st.markdown('<div class="scrollable-area">', unsafe_allow_html=True)

# Display conversation or welcome message
if not st.session_state.conversation_history:
    # Welcome message
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">🤖</div>
        <div class="welcome-title">Welcome to Data Science Assistant! 😊</div>
        <div class="welcome-text">
            I'm here to help you with Data Science, Machine Learning, AI, and related topics.<br>
            Ask me anything and I'll provide detailed, accurate answers using RAG technology.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display conversation
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="message-bubble user-bubble">
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-bubble assistant-bubble">
                {message['content']}
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # End scrollable area

# Footer (above input)
st.markdown("""
<div class="footer">
    <strong>Message RamanTech</strong><br>
    RamanTech can make mistakes. Please verify important information.
</div>
""", unsafe_allow_html=True)

# Fixed Chat Input at Bottom
st.markdown("""
<div class="chat-input-container">
    <div class="chat-input-wrapper">
""", unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Type your message here..."):
    if prompt.strip() and st.session_state.system_initialized:
        # Add user message
        user_msg = {
            'role': 'user',
            'content': prompt,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.conversation_history.append(user_msg)
        
        # Update chat title if first message
        if len(st.session_state.conversation_history) == 1:
            title = prompt[:25] + "..." if len(prompt) > 25 else prompt
            st.session_state.all_chats[st.session_state.current_chat_id]["title"] = title
        
        # Get response
        with st.spinner("Thinking..."):
            try:
                # Get RAG response
                response = st.session_state.rag_system.get_response(prompt)
                
                # Check if research needed
                response_lower = str(response).lower()
                needs_research = any(phrase in response_lower for phrase in NO_ANSWER_PHRASES)
                
                if needs_research:
                    with st.spinner("Researching additional information..."):
                        research_response = self_research.receive_and_save_research(prompt)
                        if research_response and str(research_response).strip():
                            response = research_response
                
                # Add assistant message
                assistant_msg = {
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.conversation_history.append(assistant_msg)
                
                # Update chat history
                st.session_state.all_chats[st.session_state.current_chat_id]["messages"] = st.session_state.conversation_history.copy()
                
            except Exception as e:
                error_msg = "I apologize, but I'm having trouble processing your request. Please try again."
                assistant_msg = {
                    'role': 'assistant',
                    'content': error_msg,
                    'timestamp': datetime.now().isoformat(),
                    'error': True
                }
                st.session_state.conversation_history.append(assistant_msg)
        
        st.rerun()

st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True)

# Add JavaScript for auto-scroll
st.markdown("""
<script>
// Auto-scroll to bottom
function scrollToBottom() {
    window.scrollTo(0, document.body.scrollHeight);
}

// Scroll on page load
window.addEventListener('load', scrollToBottom);

// Scroll after new message (simplified approach)
setTimeout(scrollToBottom, 100);
</script>
""", unsafe_allow_html=True)
