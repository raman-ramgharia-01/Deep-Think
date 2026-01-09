import streamlit as st
import json
import time
import traceback
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
    page_title="Deep Think AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with dark text colors
st.markdown("""
<style>
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Main container */
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    /* Chat message styling - DARK TEXT COLORS */
    .user-message {
        background-color: #e3f2fd;
        padding: 15px 20px;
        border-radius: 18px 18px 0 18px;
        margin: 10px 0 10px auto;
        max-width: 80%;
        color: #333333 !important;  /* Dark text */
        font-size: 16px;
        line-height: 1.5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 0;
        margin: 10px auto 10px 0;
        max-width: 80%;
        color: #333333 !important;  /* Dark text */
        font-size: 16px;
        line-height: 1.5;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Message content styling */
    .message-content {
        color: #333333 !important;
        margin: 0;
    }
    
    /* Typing indicator */
    .typing-indicator {
        background-color: #f8f9fa;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 0;
        margin: 10px auto 10px 0;
        max-width: 80%;
        display: inline-block;
    }
    
    .typing-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #666;
        margin: 0 3px;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-8px); }
    }
    
    /* Chat input container - Fixed at bottom */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        color: #fff;
        display: none;
        padding: 20px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    
    .chat-input-wrapper {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    /* Scrollable chat area */
    .chat-area {
        margin-bottom: 120px; /* Space for fixed input */
        padding: 20px 0;
    }
    
    /* Strong text styling */
    strong {
        color: #333333 !important;
    }
    
    /* Fix Streamlit's default text colors */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        color: #333333 !important;
    }
    
    /* Ensure all text is visible */
    * {
        color: #333333 !important;
    }
    
    /* Customize the chat input */
    div[data-testid="stChatInput"] {
        width: 100%;
    }
    textarea{
        color: #fff;
    }
    .stVerticalBlock st-emotion-cache-1353z0o eyzqfg11{
        background: #262730;
        padding: 10px;
    }
    div[data-testid="stChatInput"] > div {
        background: white;
        border: 2px solid #667eea;
        border-radius: 25px;
        padding: 5px;
    }
    
    /* Footer */
    .footer {
        display: none;
        text-align: center;
        color: #888;
        font-size: 0.9em;
        padding: 1rem;
        margin-top: 2rem;
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

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

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

# Initialize RAG system (hidden from users)
def initialize_rag_system(api_key=None):
    """Initialize RAG system"""
    try:
        rag_system = RAGSystem(api_key)
        st.session_state.system_initialized = True
        st.session_state.rag_system = rag_system
        return True
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return False

# ============================================
# MAIN CHAT INTERFACE
# ============================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Deep Think AI Assistant</h1>
    <p>Ask me anything about Data Science and Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Check if system needs initialization
if not st.session_state.system_initialized:
    # Try to initialize automatically
    with st.spinner("🔄 Initializing AI System..."):
        # Check for API key in secrets first
        api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
        
        if api_key and os.path.exists("normalize_data.joblib"):
            if initialize_rag_system(api_key):
                st.rerun()
            else:
                st.error("Failed to initialize system. Please check configuration.")
                st.stop()
        else:
            st.error("System configuration missing. Please contact administrator.")
            st.stop()

# Main chat container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Chat area with scrollable content
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

# Display chat history with proper styling
if st.session_state.conversation_history:
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                <div class="message-content">
                    <strong>You:</strong> {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if message.get('error', False):
                st.markdown(f"""
                <div class="assistant-message" style="border-left-color: #ff6b6b;">
                    <div class="message-content">
                        <strong>🤖 Assistant:</strong> {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="message-content">
                        <strong>🤖 Assistant:</strong> {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close chat-area

# Add some space before the fixed input
st.markdown('<br><br><br><br>', unsafe_allow_html=True)

# Fixed chat input at the bottom
st.markdown("""
<div class="chat-input-container">
    <div class="chat-input-wrapper">
""", unsafe_allow_html=True)

# Chat input with custom styling
if prompt := st.chat_input("Type your question here..."):
    if prompt.strip():
        # Add user message to history
        user_msg = {
            'role': 'user',
            'content': prompt,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.conversation_history.append(user_msg)
        
        # Show user message immediately
        st.markdown(f"""
        <div class="user-message">
            <div class="message-content">
                <strong>You:</strong> {prompt}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create placeholder for assistant response
        response_placeholder = st.empty()
        
        # Show typing indicator
        response_placeholder.markdown("""
        <div class="typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Get RAG response
            response = st.session_state.rag_system.get_response(prompt)
            
            # Check if research needed
            response_lower = str(response).lower()
            needs_research = any(phrase in response_lower for phrase in NO_ANSWER_PHRASES)
            
            if needs_research:
                # Update typing indicator to show researching
                response_placeholder.markdown("""
                <div class="typing-indicator">
                    🔍 Researching additional information...
                </div>
                """, unsafe_allow_html=True)
                
                # Get research response
                research_response = self_research.receive_and_save_research(prompt)
                
                if research_response and str(research_response).strip():
                    response = research_response
            
            # Clear typing indicator
            response_placeholder.empty()
            
            # Display response with typing animation
            full_response = ""
            for char in response:
                full_response += char
                response_placeholder.markdown(f"""
                <div class="assistant-message">
                    <div class="message-content">
                        <strong>🤖 Assistant:</strong> {full_response}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.01)  # Typing speed
            
            # Add to history
            assistant_msg = {
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.conversation_history.append(assistant_msg)
            
        except Exception as e:
            # Clear typing indicator
            response_placeholder.empty()
            
            error_msg = "I apologize, but I'm having trouble processing your request at the moment. Please try again."
            response_placeholder.markdown(f"""
            <div class="assistant-message">
                <div class="message-content">
                    <strong>🤖 Assistant:</strong> {error_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add error to history
            error_msg_obj = {
                'role': 'assistant',
                'content': error_msg,
                'timestamp': datetime.now().isoformat(),
                'error': True
            }
            st.session_state.conversation_history.append(error_msg_obj)

# Close the input wrapper and container
st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

# Simple footer
st.markdown("""
<div class="footer">
    <p>Powered by Deep Think AI • Ask anything about Data Science</p>
</div>
""", unsafe_allow_html=True)

# Hidden admin panel (only shown if URL parameter is set)
query_params = st.query_params
if query_params.get("admin") == "true":
    with st.expander("🔧 Admin Panel (Hidden)", expanded=False):
        st.write("**System Status:**")
        st.write(f"- Initialized: {st.session_state.system_initialized}")
        st.write(f"- Total Messages: {len(st.session_state.conversation_history)}")
        
        if st.session_state.rag_system and st.session_state.rag_system.df is not None:
            st.write(f"- Documents Loaded: {st.session_state.rag_system.df.shape[0]}")
        
        if st.button("Clear Chat History"):
            st.session_state.conversation_history = []
            st.rerun()
        
        if st.button("Export Chat"):
            if st.session_state.conversation_history:
                export_data = {
                    'export_date': datetime.now().isoformat(),
                    'conversation': st.session_state.conversation_history
                }
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    "Download JSON",
                    json_str,
                    f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
