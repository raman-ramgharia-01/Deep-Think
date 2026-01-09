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
    def __init__(self):
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
            # Get API key from Streamlit secrets
            api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
            
            if not api_key:
                return "API key not found. Please set GROQ_API_KEY in secrets or environment variables."
            
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
@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with caching"""
    st.info("🔄 Initializing RAG System...")
    rag_system = RAGSystem()
    if rag_system.df is not None:
        st.success("✅ RAG System initialized successfully!")
        st.session_state.system_initialized = True
    else:
        st.error("❌ Failed to initialize RAG System")
        st.session_state.system_initialized = False
    return rag_system

# Sidebar
with st.sidebar:
    st.title("⚙️ System Control")
    
    st.markdown("---")
    
    # Initialize system button
    if not st.session_state.system_initialized:
        if st.button("🚀 Initialize RAG System", use_container_width=True, type="primary"):
            with st.spinner("Loading embeddings and models..."):
                st.session_state.rag_system = initialize_rag_system()
                st.rerun()
    else:
        st.success("✅ System Initialized")
        if st.button("🔄 Re-initialize System", use_container_width=True):
            with st.spinner("Reloading system..."):
                st.session_state.rag_system = initialize_rag_system()
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
    
    if st.session_state.system_initialized and st.session_state.rag_system:
        st.success("✅ RAG: Ready")
        st.success(f"📊 Data: {st.session_state.rag_system.df.shape[0]} documents")
    else:
        st.error("❌ RAG: Not Initialized")
    
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

# Main interface
st.title("🤖 Deep Think - RAG Chat Assistant")
st.markdown("""
<div class="info-box">
This intelligent assistant combines RAG (Retrieval-Augmented Generation) with self-research capabilities. 
Ask questions about Data Science, and the system will retrieve relevant information from its knowledge base.
</div>
""", unsafe_allow_html=True)

# Check if system is initialized
if not st.session_state.system_initialized:
    st.warning("""
    ⚠️ **System not initialized!**
    
    Please click **'Initialize RAG System'** in the sidebar to start.
    
    Requirements:
    1. `normalize_data.joblib` file must be in the project directory
    2. GROQ_API_KEY must be set in secrets or environment variables
    """)
    
    # Show system requirements
    with st.expander("🔧 System Requirements", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 File Check")
            if os.path.exists("normalize_data.joblib"):
                st.success("✅ normalize_data.joblib found")
                try:
                    file_size = os.path.getsize("normalize_data.joblib") / (1024*1024)
                    st.write(f"Size: {file_size:.2f} MB")
                except:
                    pass
            else:
                st.error("❌ normalize_data.joblib not found")
        
        with col2:
            st.subheader("🔑 API Key Check")
            api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
            if api_key:
                st.success("✅ GROQ_API_KEY found")
                st.write(f"Key: {api_key[:10]}...")
            else:
                st.error("❌ GROQ_API_KEY not found")
    
    st.stop()

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
        # Clear cache but keep conversation
        st.cache_resource.clear()
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>🤖 <b>Deep Think RAG Assistant</b> | Powered by Sentence Transformers, Groq API, and Streamlit</p>
    <p>⚡ Real-time retrieval augmented generation with self-research capabilities</p>
</div>
""", unsafe_allow_html=True)
