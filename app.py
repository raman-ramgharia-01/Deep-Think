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
    page_title="Deep Think - RAG Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .input-box {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

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
        self.api_key = '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'
        
        # Initialize model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load data
        try:
            self.df = joblib.load('normalize_data.joblib')
            st.success(f"✅ Data loaded successfully! Shape: {self.df.shape}")
        except FileNotFoundError:
            st.error("❌ Error: 'normalize_data.joblib' not found")
            self.df = None
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            self.df = None
    
    def analyze_with_groq(self, text_data):
        """Send text to Groq API and get response"""
        try:
            # Use the stored API key
            api_key = self.api_key or st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
            
            if not api_key:
                return "API key not found. Please set GROQ_API_KEY."
            
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
            st.error(f"Error calling Groq API: {e}")
            return "Sorry, I encountered an error while processing your request."
    
    def get_response(self, user_query):
        """Main function to process query and return response"""
        if self.df is None or len(self.df) == 0:
            return "System not properly initialized. Please check data files."
        
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
        
        # Save prompt (optional)
        try:
            with open("prompt.txt", 'w', encoding='utf-8') as f:
                f.write(rag_prompt)
        except:
            pass
        
        # Get response from Groq
        response = self.analyze_with_groq(rag_prompt)
        
        return response

# Initialize RAG system
def initialize_rag_system(api_key=None):
    """Initialize RAG system"""
    st.info("🔄 Initializing RAG System...")
    rag_system = RAGSystem(api_key)
    if rag_system.df is not None:
        st.success("✅ RAG System initialized successfully!")
        st.session_state.system_initialized = True
        st.session_state.rag_system = rag_system
    else:
        st.error("❌ Failed to initialize RAG System")
        st.session_state.system_initialized = False
        st.session_state.rag_system = None

# Main initialization interface
if not st.session_state.system_initialized:
    st.title("🤖 Deep Think - Setup Required")
    
    st.markdown("""
    <div class="info-box">
    <h3>🚀 Welcome to Deep Think RAG Assistant!</h3>
    <p>Before we start, we need to set up the system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 File Check")
        st.write("Checking for required data files...")
        
        if os.path.exists("normalize_data.joblib"):
            st.success("✅ normalize_data.joblib found")
            try:
                file_size = os.path.getsize("normalize_data.joblib") / (1024*1024)
                st.write(f"Size: {file_size:.2f} MB")
            except:
                pass
        else:
            st.error("❌ normalize_data.joblib not found")
            st.info("Please upload or place this file in the project directory.")
            
            # File upload option
            uploaded_file = st.file_uploader("Or upload normalize_data.joblib", type=['joblib'])
            if uploaded_file:
                with open("normalize_data.joblib", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("✅ File uploaded successfully!")
                st.rerun()
    
    with col2:
        st.subheader("🔑 API Configuration")
        
        # Check existing API keys
        secrets_key = st.secrets.get("GROQ_API_KEY", "")
        env_key = os.environ.get("GROQ_API_KEY", "")
        
        if secrets_key:
            st.success("✅ GROQ_API_KEY found in Streamlit secrets")
            st.session_state.api_key = secrets_key
        elif env_key:
            st.success("✅ GROQ_API_KEY found in environment variables")
            st.session_state.api_key = env_key
        else:
            st.warning("⚠️ No API key found in secrets or environment")
            
            # Manual API key input
            st.subheader("Enter API Key Manually")
            api_key_input = st.text_input(
                "GROQ API Key",
                type="password",
                placeholder="Enter your GROQ API key here...",
                help="Get your API key from https://console.groq.com"
            )
            
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.success("✅ API key entered")
    
    st.markdown("---")
    
    # Initialize button
    st.subheader("🚀 Initialize System")
    
    # Check if we can initialize
    can_initialize = os.path.exists("normalize_data.joblib") and st.session_state.api_key
    
    if can_initialize:
        if st.button("🎯 Initialize RAG System Now", type="primary", use_container_width=True):
            with st.spinner("Loading embeddings and initializing models..."):
                initialize_rag_system(st.session_state.api_key)
                if st.session_state.system_initialized:
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
    else:
        st.error("❌ Cannot initialize - missing requirements:")
        if not os.path.exists("normalize_data.joblib"):
            st.write("- ❌ normalize_data.joblib file not found")
        if not st.session_state.api_key:
            st.write("- ❌ GROQ_API_KEY not provided")
    
    # Debug info
    with st.expander("🔧 Technical Details", expanded=False):
        st.write("**Current Status:**")
        st.write(f"- Data file exists: {os.path.exists('normalize_data.joblib')}")
        st.write(f"- API key provided: {bool(st.session_state.api_key)}")
        st.write(f"- System initialized: {st.session_state.system_initialized}")
        
        st.write("**Directory Contents:**")
        try:
            files = os.listdir(".")
            st.write(f"Files in current directory: {len(files)}")
            for file in files[:10]:  # Show first 10 files
                st.write(f"- {file}")
        except:
            st.write("Cannot list directory contents")
    
    # Stop execution here - don't show the chat interface yet
    st.stop()

# ============================================
# MAIN CHAT INTERFACE (only shows after initialization)
# ============================================

# Sidebar (only shows after initialization)
with st.sidebar:
    st.title("⚙️ System Control")
    
    st.markdown("---")
    
    # System status
    st.success("✅ System Initialized")
    
    if st.button("🔄 Re-initialize System", use_container_width=True):
        # Reset and show setup again
        st.session_state.system_initialized = False
        st.session_state.rag_system = None
        st.rerun()
    
    st.markdown("---")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.checkbox("🔍 Debug Mode", value=False)
    if st.session_state.debug_mode:
        st.warning("Debug mode enabled")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()
    
    st.markdown("---")
    
    # System Info
    st.subheader("📊 System Status")
    
    if st.session_state.rag_system and st.session_state.rag_system.df is not None:
        st.success("✅ RAG: Ready")
        st.success(f"📊 Data: {st.session_state.rag_system.df.shape[0]} documents")
    else:
        st.error("❌ RAG: Error")
    
    try:
        research_status = "✅ Ready" if self_research else "❌ Error"
        st.success(f"🔍 Research: {research_status}")
    except:
        st.error("❌ Research: Module Error")
    
    st.write(f"💬 Messages: {len(st.session_state.conversation_history)}")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    if st.button("📋 Copy Last Response", use_container_width=True):
        if st.session_state.conversation_history:
            last_assistant = next(
                (msg['content'] for msg in reversed(st.session_state.conversation_history) 
                 if msg['role'] == 'assistant' and not msg.get('error', False)),
                None
            )
            if last_assistant:
                st.code(last_assistant)
                st.success("Copied!")
            else:
                st.warning("No assistant response found")
    
    if st.button("📈 View Data Stats", use_container_width=True):
        if st.session_state.rag_system and st.session_state.rag_system.df is not None:
            st.write("**DataFrame Info:**")
            st.write(f"Shape: {st.session_state.rag_system.df.shape}")
            st.write(f"Columns: {list(st.session_state.rag_system.df.columns)}")
            st.write("**Sample Data:**")
            st.dataframe(st.session_state.rag_system.df.head(3))

# Main chat interface
st.title("🤖 Deep Think - RAG Chat Assistant")
st.markdown("""
<div class="info-box">
✅ **System Ready!** Ask me anything about Data Science.
I'll retrieve relevant information and provide detailed answers.
</div>
""", unsafe_allow_html=True)

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(message['content'])
                if 'timestamp' in message:
                    st.caption(f"🕒 {message['timestamp']}")
        else:
            with st.chat_message("assistant"):
                if message.get('error', False):
                    st.error(message['content'])
                else:
                    st.markdown(message['content'])
                if 'timestamp' in message:
                    st.caption(f"🕒 {message['timestamp']}")

# Chat input
if prompt := st.chat_input("Ask me anything about Data Science..."):
    if prompt.strip():
        # Add user message
        user_msg = {
            'role': 'user',
            'content': prompt,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.conversation_history.append(user_msg)
        
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"🕒 {user_msg['timestamp']}")
        
        # Process with assistant
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                # Step 1: Get RAG response
                status_placeholder.info("🔍 **Retrieving relevant information...**")
                response = st.session_state.rag_system.get_response(prompt)
                
                if st.session_state.debug_mode:
                    with st.expander("📋 Debug: RAG Response", expanded=False):
                        st.write(f"**Raw Response:**")
                        st.code(response)
                
                # Step 2: Check if research needed
                response_lower = str(response).lower()
                needs_research = any(phrase in response_lower for phrase in NO_ANSWER_PHRASES)
                
                if needs_research:
                    status_placeholder.warning("📚 **Researching further...**")
                    
                    # Get research response
                    research_response = self_research.receive_and_save_research(prompt)
                    
                    if research_response and str(research_response).strip():
                        response = research_response
                        status_placeholder.success("✅ **New information added to knowledge base!**")
                    else:
                        status_placeholder.warning("⚠️ **No additional information found**")
                
                # Display final response
                message_placeholder.markdown(response)
                status_placeholder.empty()
                
                # Add to history
                assistant_msg = {
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
                st.session_state.conversation_history.append(assistant_msg)
                
                st.caption(f"🕒 {assistant_msg['timestamp']}")
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                
                if st.session_state.debug_mode:
                    error_details = traceback.format_exc()
                    st.error("### 🔍 Debug Information")
                    st.code(error_details)
                    st.write(f"**Error Type:** {type(e).__name__}")
                    st.write(f"**Prompt:** {prompt}")
                
                message_placeholder.error(error_msg)
                status_placeholder.empty()
                
                # Add error to history
                error_msg_obj = {
                    'role': 'assistant',
                    'content': error_msg,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'error': True
                }
                st.session_state.conversation_history.append(error_msg_obj)

# Debug panel
if st.session_state.debug_mode and st.session_state.system_initialized:
    with st.expander("🔬 Advanced Debug Panel", expanded=False):
        st.subheader("📊 Data Exploration")
        
        if st.session_state.rag_system and st.session_state.rag_system.df is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Documents", st.session_state.rag_system.df.shape[0])
            
            with col2:
                st.metric("Embedding Dimensions", 
                         len(st.session_state.rag_system.df['embedding'].iloc[0]) 
                         if len(st.session_state.rag_system.df) > 0 else 0)
            
            with col3:
                avg_text_len = st.session_state.rag_system.df['text'].str.len().mean()
                st.metric("Avg Text Length", f"{avg_text_len:.0f} chars")
            
            # Show sample data
            st.subheader("📄 Sample Documents")
            st.dataframe(st.session_state.rag_system.df[['text']].head(5))
        
        st.subheader("🎯 Test Queries")
        test_queries = [
            "What is machine learning?",
            "Explain neural networks",
            "What is gradient descent?",
            "How does K-means clustering work?"
        ]
        
        for query in test_queries:
            if st.button(f"Test: {query}"):
                with st.spinner(f"Testing: {query}"):
                    try:
                        test_response = st.session_state.rag_system.get_response(query)
                        st.write(f"**Query:** {query}")
                        st.write(f"**Response:** {test_response[:200]}...")
                        st.divider()
                    except Exception as e:
                        st.error(f"Test failed: {e}")

# Export and utilities
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Export Chat History", use_container_width=True):
        if st.session_state.conversation_history:
            export_data = {
                'export_date': datetime.now().isoformat(),
                'system': 'Deep Think RAG Assistant',
                'total_messages': len(st.session_state.conversation_history),
                'conversation': st.session_state.conversation_history
            }
            
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📥 Download JSON",
                json_str,
                f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        else:
            st.warning("No chat history to export")

with col2:
    if st.button("📊 View Statistics", use_container_width=True):
        if st.session_state.conversation_history:
            user_msgs = [m for m in st.session_state.conversation_history if m['role'] == 'user']
            assistant_msgs = [m for m in st.session_state.conversation_history if m['role'] == 'assistant']
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("User Messages", len(user_msgs))
            with col_b:
                st.metric("Assistant Messages", len(assistant_msgs))
        else:
            st.info("No statistics available")

with col3:
    if st.button("🔄 Refresh Session", use_container_width=True):
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>🤖 <b>Deep Think RAG Assistant</b> | Powered by Sentence Transformers, Groq API, and Streamlit</p>
    <p>API Key: {st.session_state.api_key[:10]}...</p>
</div>
""", unsafe_allow_html=True)
