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
    initial_sidebar_state="collapsed"
)

# Custom CSS for the exact UI
st.markdown("""
<style>
    /* Hide all Streamlit default elements */
    #MainMenu, header, footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Main container */
    .main-container {
        display: flex;
        min-height: 100vh;
        background-color: #f5f5f5;
    }
    
    /* Left sidebar - Recent chats */
    .sidebar {
        width: 280px;
        background: white;
        padding: 25px;
        border-right: 1px solid #e0e0e0;
        display: flex;
        flex-direction: column;
    }
    
    .logo {
        font-size: 24px;
        font-weight: 700;
        color: #000;
        margin-bottom: 10px;
    }
    
    .subtitle {
        font-size: 14px;
        color: #666;
        margin-bottom: 40px;
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
        letter-spacing: 1px;
        margin-bottom: 20px;
    }
    
    .chat-item {
        padding: 12px 15px;
        margin-bottom: 10px;
        border-radius: 10px;
        cursor: pointer;
        font-size: 14px;
        color: #333;
        transition: all 0.2s;
        background: #f8f9fa;
    }
    
    .chat-item:hover {
        background: #e9ecef;
    }
    
    .chat-item.active {
        background: #007bff;
        color: white;
    }
    
    /* Main chat area */
    .chat-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        background: white;
    }
    
    .chat-header {
        padding: 25px;
        border-bottom: 1px solid #e0e0e0;
        background: white;
    }
    
    .chat-header h1 {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #000;
    }
    
    .chat-header p {
        font-size: 16px;
        color: #666;
        margin: 0;
    }
    
    /* Messages area */
    .messages-container {
        flex: 1;
        padding: 25px;
        overflow-y: auto;
        background: white;
    }
    
    .message {
        margin-bottom: 25px;
        max-width: 70%;
    }
    
    .user-message {
        margin-left: auto;
        background: #007bff;
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 0 20px;
    }
    
    .assistant-message {
        background: #f8f9fa;
        color: #333;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 0;
        border-left: 4px solid #007bff;
    }
    
    .message-time {
        font-size: 12px;
        color: #999;
        margin-top: 5px;
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center;
        padding: 40px 20px;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .welcome-title {
        font-size: 28px;
        font-weight: 600;
        color: #333;
        margin-bottom: 15px;
    }
    
    .welcome-subtitle {
        font-size: 16px;
        color: #666;
        line-height: 1.6;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        padding: 15px 20px;
        background: #f8f9fa;
        border-radius: 20px 20px 20px 0;
        border-left: 4px solid #007bff;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #888;
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.6; }
        30% { transform: translateY(-5px); opacity: 1; }
    }
    
    /* Input area */
    .input-container {
        padding: 25px;
        border-top: 1px solid #e0e0e0;
        background: white;
        position: sticky;
        bottom: 0;
    }
    
    .input-wrapper {
        max-width: 600px;
        margin: 0 auto;
        position: relative;
    }
    
    .chat-input {
        width: 100%;
        padding: 15px 50px 15px 20px;
        border: 2px solid #e0e0e0;
        border-radius: 25px;
        font-size: 16px;
        outline: none;
        transition: border-color 0.2s;
    }
    
    .chat-input:focus {
        border-color: #007bff;
    }
    
    .send-button {
        position: absolute;
        right: 15px;
        top: 50%;
        transform: translateY(-50%);
        background: #007bff;
        color: white;
        border: none;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 16px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #888;
        font-size: 13px;
        border-top: 1px solid #e0e0e0;
        background: white;
    }
    
    .footer a {
        color: #007bff;
        text-decoration: none;
    }
    
    /* Scrollbar styling */
    .messages-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .messages-container::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    .messages-container::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    .messages-container::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* New chat button */
    .new-chat-btn {
        background: #007bff;
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 10px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin-top: auto;
        transition: background-color 0.2s;
    }
    
    .new-chat-btn:hover {
        background: #0056b3;
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
    st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if 'all_chats' not in st.session_state:
    st.session_state.all_chats = {}

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
        initialize_rag_system(api_key)

# Store current chat in all chats
if st.session_state.current_chat_id not in st.session_state.all_chats:
    st.session_state.all_chats[st.session_state.current_chat_id] = {
        'id': st.session_state.current_chat_id,
        'title': f"Chat {len(st.session_state.all_chats) + 1}",
        'messages': st.session_state.conversation_history.copy(),
        'created_at': datetime.now().strftime("%H:%M")
    }

# Function to create new chat
def create_new_chat():
    """Create a new chat session"""
    # Save current chat
    if st.session_state.conversation_history:
        # Create title from first message
        first_msg = st.session_state.conversation_history[0]['content'] if st.session_state.conversation_history else "New Chat"
        title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
        
        st.session_state.all_chats[st.session_state.current_chat_id] = {
            'id': st.session_state.current_chat_id,
            'title': title,
            'messages': st.session_state.conversation_history.copy(),
            'created_at': datetime.now().strftime("%H:%M")
        }
    
    # Create new chat
    new_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_chat_id = new_chat_id
    st.session_state.conversation_history = []
    
    # Initialize new chat entry
    st.session_state.all_chats[new_chat_id] = {
        'id': new_chat_id,
        'title': f"Chat {len(st.session_state.all_chats) + 1}",
        'messages': [],
        'created_at': datetime.now().strftime("%H:%M")
    }

# Function to load chat
def load_chat(chat_id):
    """Load a specific chat"""
    if chat_id in st.session_state.all_chats:
        st.session_state.current_chat_id = chat_id
        st.session_state.conversation_history = st.session_state.all_chats[chat_id]['messages'].copy()

# ============================================
# UI COMPONENTS
# ============================================

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Left Sidebar
with st.container():
    st.markdown("""
    <div class="sidebar">
        <div class="logo">Data Science Assistant</div>
        <div class="subtitle">Powered by RAG + Llama 3.3</div>
        
        <div class="divider"></div>
        
        <div class="section-title">Recent Chats</div>
    """, unsafe_allow_html=True)
    
    # Display recent chats
    recent_chats = list(st.session_state.all_chats.values())[-5:]  # Last 5 chats
    for chat in reversed(recent_chats):
        chat_title = chat['title']
        chat_time = chat['created_at']
        is_active = chat['id'] == st.session_state.current_chat_id
        
        st.markdown(f"""
        <div class="chat-item {'active' if is_active else ''}" onclick="loadChat('{chat['id']}')">
            {chat_title}<br>
            <small>{chat_time}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # New Chat Button
    st.markdown("""
        <button class="new-chat-btn" onclick="newChat()">
            <span>+</span> New Chat
        </button>
    </div>
    """, unsafe_allow_html=True)

# Main Chat Area
st.markdown("""
<div class="chat-container">
    <div class="chat-header">
        <h1>How can I help you?</h1>
        <p>Ask me anything about Data Science, Machine Learning, or related topics</p>
    </div>
    
    <div class="messages-container" id="messages-container">
""", unsafe_allow_html=True)

# Display messages if any
if st.session_state.conversation_history:
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="message user-message">
                {message['content']}
                <div class="message-time">You • Now</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message assistant-message">
                {message['content']}
                <div class="message-time">Assistant • Now</div>
            </div>
            """, unsafe_allow_html=True)
else:
    # Welcome message
    st.markdown("""
    <div class="welcome-message">
        <div class="welcome-title">Welcome to Data Science Assistant! 🤖</div>
        <div class="welcome-subtitle">
            I'm here to help you with Data Science, Machine Learning, AI, and related topics.<br>
            Ask me anything and I'll provide detailed, accurate answers using RAG technology.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # Close messages-container

# Chat input area
st.markdown("""
<div class="input-container">
    <div class="input-wrapper">
        <form id="chat-form">
            <input type="text" class="chat-input" id="chat-input" placeholder="Type your message here..." autocomplete="off">
            <button type="submit" class="send-button">→</button>
        </form>
    </div>
</div>

<div class="footer">
    <strong>Message RamanTech</strong><br>
    RamanTech can make mistakes. Please verify important information.
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # Close chat-container
st.markdown("</div>", unsafe_allow_html=True)  # Close main-container

# JavaScript for interactivity
st.markdown("""
<script>
// Function to load a chat
function loadChat(chatId) {
    // In Streamlit, we need to use streamlit:setComponentValue to communicate with Python
    const event = new CustomEvent('streamlit:setComponentValue', {
        detail: {type: 'load_chat', chatId: chatId}
    });
    window.parent.document.dispatchEvent(event);
}

// Function to create new chat
function newChat() {
    const event = new CustomEvent('streamlit:setComponentValue', {
        detail: {type: 'new_chat'}
    });
    window.parent.document.dispatchEvent(event);
}

// Handle form submission
document.getElementById('chat-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (message) {
        // Send message to Streamlit
        const event = new CustomEvent('streamlit:setComponentValue', {
            detail: {type: 'send_message', message: message}
        });
        window.parent.document.dispatchEvent(event);
        input.value = '';
    }
});

// Focus on input
document.getElementById('chat-input').focus();

// Auto-scroll to bottom
function scrollToBottom() {
    const container = document.getElementById('messages-container');
    container.scrollTop = container.scrollHeight;
}

// Scroll on load
window.addEventListener('load', scrollToBottom);
</script>
""", unsafe_allow_html=True)

# Handle JavaScript events
if 'streamlit:setComponentValue' in st.session_state:
    event_data = st.session_state['streamlit:setComponentValue']
    
    if event_data.get('type') == 'send_message' and 'message' in event_data:
        prompt = event_data['message']
        
        if prompt.strip() and st.session_state.system_initialized:
            # Add user message
            user_msg = {
                'role': 'user',
                'content': prompt,
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.conversation_history.append(user_msg)
            
            # Update chat title with first message
            if len(st.session_state.conversation_history) == 1:  # First message
                title = prompt[:30] + "..." if len(prompt) > 30 else prompt
                st.session_state.all_chats[st.session_state.current_chat_id]['title'] = title
            
            try:
                # Get RAG response
                response = st.session_state.rag_system.get_response(prompt)
                
                # Check if research needed
                response_lower = str(response).lower()
                needs_research = any(phrase in response_lower for phrase in NO_ANSWER_PHRASES)
                
                if needs_research:
                    # Get research response
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
                
                # Update chat in all_chats
                st.session_state.all_chats[st.session_state.current_chat_id]['messages'] = st.session_state.conversation_history.copy()
                
            except Exception as e:
                error_msg = "I apologize, but I'm having trouble processing your request."
                assistant_msg = {
                    'role': 'assistant',
                    'content': error_msg,
                    'timestamp': datetime.now().isoformat(),
                    'error': True
                }
                st.session_state.conversation_history.append(assistant_msg)
            
            st.rerun()
    
    elif event_data.get('type') == 'new_chat':
        create_new_chat()
        st.rerun()
    
    elif event_data.get('type') == 'load_chat' and 'chatId' in event_data:
        load_chat(event_data['chatId'])
        st.rerun()

# For Streamlit to capture JavaScript events
st.components.v1.html("""
<div id="event-handler"></div>
<script>
// Listen for events from our JavaScript
document.addEventListener('streamlit:setComponentValue', function(e) {
    // This would normally communicate with Streamlit
    // Since we can't directly modify session state from JS, we use a different approach
    window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        data: e.detail
    }, '*');
});

// Listen for messages from parent (Streamlit)
window.addEventListener('message', function(event) {
    if (event.data && event.data.type === 'streamlit:setComponentValue') {
        document.dispatchEvent(new CustomEvent('streamlit:setComponentValue', {
            detail: event.data.data
        }));
    }
});
</script>
""", height=0)

# Alternative approach using query parameters for navigation
query_params = st.query_params
if 'new_chat' in query_params:
    create_new_chat()
    st.query_params.clear()
    st.rerun()

if 'load_chat' in query_params:
    load_chat(query_params['load_chat'])
    st.query_params.clear()
    st.rerun()

# Check if system is ready
if not st.session_state.system_initialized:
    st.error("System initialization failed. Please check your configuration.")
    st.stop()
