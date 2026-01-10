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
    page_title="RamanTech - Data Science Assistant",
    page_icon="https://cdn-icons-png.flaticon.com/512/7266/7266268.png",
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
        color: #fff;
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
        color: #fff;
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
        background: #000;
        padding: 20px;
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
        bottom: 10px;
        left: 0;
        right: 0;
        background: #000;
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
        
        # Load data with error handling
        try:
            self.df = joblib.load('normalize_data.joblib')
            # st.success(f"✅ Data loaded: {len(self.df)} documents")
            
            # Debug: Check data structure
            if 'embedding' in self.df.columns:
                # st.info(f"Embedding column found. Sample type: {type(self.df['embedding'].iloc[0])}")
                
                # Convert embeddings to numpy arrays if needed
                if isinstance(self.df['embedding'].iloc[0], list):
                    self.df['embedding'] = self.df['embedding'].apply(np.array)
                    st.info("Converted embeddings from lists to numpy arrays")
                
        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")
            self.df = None
    
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
            st.error(f"API Error: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def get_response(self, user_query):
        """Main function to process query and return response"""
        if self.df is None or len(self.df) == 0:
            return "System not properly initialized. Please check data files."
        
        try:
            # Get query embedding
            user_embedding = self.model.encode(user_query)
            reshaped_user_embedding = user_embedding.reshape(1, -1)
            normalized_user_embedding = normalize(reshaped_user_embedding, norm='max')[0]
            
            # Convert embeddings to proper format
            try:
                # Check if embeddings need conversion
                sample_embedding = self.df['embedding'].iloc[0]
                
                if not isinstance(sample_embedding, np.ndarray):
                    # Try to convert to numpy array
                    self.df['embedding'] = self.df['embedding'].apply(
                        lambda x: np.array(x) if isinstance(x, list) else x
                    )
                
                # Stack embeddings for similarity calculation
                embedding_matrix = np.vstack(self.df['embedding'].values)
                
            except Exception as e:
                st.error(f"Error processing embeddings: {e}")
                return "Error processing the knowledge base. Please check the data format."
            
            # Calculate similarities with error handling
            try:
                similarities = cosine_similarity(
                    normalized_user_embedding.reshape(1, -1), 
                    embedding_matrix
                )
                
                # Get top results
                top_results = min(3, len(self.df))
                top_chunks = similarities[0].argsort()[-top_results:][::-1]
                
            except Exception as e:
                st.error(f"Error calculating similarities: {e}")
                return "Error finding relevant information. Please try a different question."
            
            # Build retrieved context
            retrieved_context = ""
            for idx in top_chunks:
                if 'text' in self.df.columns and idx < len(self.df):
                    retrieved_context += self.df.iloc[idx]['text'] + "\n\n"
            
            # Create RAG prompt
            rag_prompt = f"""Context:
{retrieved_context}

Based on the context above, answer this question: {user_query}

If the context doesn't contain relevant information, say "I don't have enough information in the provided context to answer this question."

Answer:"""
            
            # Save prompt for debugging
            try:
                with open("last_prompt.txt", "w", encoding="utf-8") as f:
                    f.write(rag_prompt)
            except:
                pass
            
            # Get response from Groq
            response = self.analyze_with_groq(rag_prompt)
            
            return response
            
        except Exception as e:
            st.error(f"Error in RAG processing: {e}")
            return "I encountered an error while processing your question. Please try again."

# Initialize RAG system
def initialize_rag_system(api_key=None):
    """Initialize RAG system"""
    try:
        rag_system = RAGSystem(api_key)
        if rag_system.df is not None and len(rag_system.df) > 0:
            st.session_state.system_initialized = True
            st.session_state.rag_system = rag_system
            return True
        else:
            st.error("Failed to load data or data is empty")
            return False
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return False

# Auto-initialize system
if not st.session_state.system_initialized:
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    
    if not api_key:
        st.error("❌ GROQ_API_KEY not found. Please add it to Streamlit secrets or environment variables.")
        st.info("Add to .streamlit/secrets.toml: GROQ_API_KEY = 'your_key_here'")
    elif not os.path.exists("normalize_data.joblib"):
        st.error("❌ normalize_data.joblib not found. Please ensure this file exists in your project directory.")
    else:
        with st.spinner("🔄 Initializing RAG System..."):
            if initialize_rag_system(api_key):
                # st.success("✅ System initialized successfully!")
                pass
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
        <span>Powered by RAG • RamanTech </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Recent Chats Section
    st.markdown('<div class="section-title">CHATS</div>', unsafe_allow_html=True)
    
    # Display recent chats
    for chat_id, chat_data in list(st.session_state.all_chats.items())[-5:]:
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
    
    # System status in sidebar
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.session_state.system_initialized:
        # st.success("✅ System Ready")
        if st.session_state.rag_system and st.session_state.rag_system.df is not None:
            st.info(f"📚 {len(st.session_state.rag_system.df)} documents loaded")
    else:
        st.error("❌ System Not Ready")

# ============================================
# MAIN AREA - Right Column
# ============================================

# Check if system is ready
if not st.session_state.system_initialized:
    st.error("""
    ## ⚠️ System Not Initialized
    
    Please ensure:
    1. `normalize_data.joblib` exists in the project directory
    2. GROQ_API_KEY is set in Streamlit secrets or environment variables
    
    Check the sidebar for specific error messages.
    """)
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">RamanTech</div>
    <div class="main-subtitle">How can I help you?</div>
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
    if prompt.strip():
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
                            response = research_response[0] if research_response[1] is None else research_response[0]
                            research_data = {
                                'text': f'{user_query}: {research_response}'
                            }
                            
                            with open('research_responses.json', 'w') as f:
                                json.dump(research_data, f, indent=4)
                
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
                st.error(f"Error: {e}")
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

// Scroll after new message
setTimeout(scrollToBottom, 100);
</script>
""", unsafe_allow_html=True)

# Debug panel (hidden by default)
# with st.expander("🔧 Debug Info", expanded=False):
#     if st.session_state.rag_system and st.session_state.rag_system.df is not None:
#         st.write("**Data Information:**")
#         st.write(f"Number of documents: {len(st.session_state.rag_system.df)}")
#         st.write(f"Columns: {list(st.session_state.rag_system.df.columns)}")
        
#         if len(st.session_state.rag_system.df) > 0:
#             st.write("**Sample embedding type:**")
#             sample = st.session_state.rag_system.df['embedding'].iloc[0]
#             st.write(f"Type: {type(sample)}")
#             st.write(f"Shape: {sample.shape if hasattr(sample, 'shape') else 'N/A'}")
            
#         if st.button("Test Embedding Conversion"):
#             try:
#                 embedding_matrix = np.vstack(st.session_state.rag_system.df['embedding'].values)
#                 # st.success(f"✅ Success! Embedding matrix shape: {embedding_matrix.shape}")
#             except Exception as e:
#                 st.error(f"❌ Failed: {e}")
