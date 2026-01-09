import streamlit as st
import json
import time
import traceback
from datetime import datetime
from rag_system import rag_system
from ResearchSystem import self_research

# Set page config
st.set_page_config(
    page_title="Chat Assistant",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

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

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with debug toggle
with st.sidebar:
    st.title("🔧 Settings")
    
    st.markdown("---")
    
    # Debug toggle
    st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=False)
    
    if st.session_state.debug_mode:
        st.warning("Debug mode enabled - errors will be displayed")
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()
    
    st.markdown("---")
    
    # Module status
    st.subheader("📦 Module Status")
    
    # Check if modules are loaded
    try:
        rag_module = rag_system.__name__ if hasattr(rag_system, '__name__') else "Loaded"
        st.success(f"✅ RAG System: {rag_module}")
    except:
        st.error("❌ RAG System: Not loaded")
    
    try:
        research_module = self_research.__name__ if hasattr(self_research, '__name__') else "Loaded"
        st.success(f"✅ Research System: {research_module}")
    except:
        st.error("❌ Research System: Not loaded")
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 Stats")
    st.write(f"Messages: {len(st.session_state.conversation_history)}")

# Main interface
st.title("🤖 Chat Assistant")
st.markdown("---")

# Display chat history
for message in st.session_state.conversation_history:
    if message['role'] == 'user':
        with st.chat_message("user"):
            st.markdown(message['content'])
            if 'timestamp' in message:
                st.caption(message['timestamp'])
    else:
        with st.chat_message("assistant"):
            if message.get('error', False):
                st.error(message['content'])
            else:
                st.markdown(message['content'])
            if 'timestamp' in message:
                st.caption(message['timestamp'])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
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
            st.caption(user_msg['timestamp'])
        
        # Process with assistant
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Test RAG system first
                message_placeholder.markdown("🤔 *Thinking...*")
                
                # Get RAG response
                response = rag_system.get_response(prompt)
                
                # Debug info
                if st.session_state.debug_mode:
                    st.info(f"RAG Response Type: {type(response)}")
                    st.info(f"RAG Response Length: {len(str(response))}")
                
                # Check if response is valid
                if not response or str(response).strip() == "":
                    raise ValueError("Empty response from RAG system")
                
                # Check if research needed
                response_lower = str(response).lower()
                needs_research = any(phrase in response_lower for phrase in NO_ANSWER_PHRASES)
                
                if needs_research:
                    message_placeholder.markdown("🔍 *Researching...*")
                    
                    # Get research response
                    research_response = self_research.receive_and_save_research(prompt)
                    
                    # Debug info
                    if st.session_state.debug_mode:
                        st.info(f"Research Response Type: {type(research_response)}")
                        st.info(f"Research Response Length: {len(str(research_response))}")
                    
                    if research_response and str(research_response).strip():
                        response = research_response
                        st.success("✅ Added new information to knowledge base")
                    else:
                        st.warning("⚠️ Research returned no additional information")
                
                # Display response
                message_placeholder.markdown(response)
                
                # Add to history
                assistant_msg = {
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
                st.session_state.conversation_history.append(assistant_msg)
                
                st.caption(assistant_msg['timestamp'])
                
            except Exception as e:
                # Show detailed error in debug mode
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                
                if st.session_state.debug_mode:
                    error_details = traceback.format_exc()
                    st.error("### Debug Information")
                    st.code(error_details)
                    
                    # Show more context
                    st.error("### Error Context")
                    st.write(f"**Error Type:** {type(e).__name__}")
                    st.write(f"**Prompt:** {prompt}")
                    st.write(f"**RAG Module:** {rag_system.__class__ if hasattr(rag_system, '__class__') else 'Unknown'}")
                    st.write(f"**Research Module:** {self_research.__class__ if hasattr(self_research, '__class__') then 'Unknown'}")
                
                message_placeholder.error(error_msg)
                
                # Add error to history
                error_msg_obj = {
                    'role': 'assistant',
                    'content': error_msg,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'error': True
                }
                st.session_state.conversation_history.append(error_msg_obj)

# Debug section
if st.session_state.debug_mode:
    with st.expander("🔍 Debug Information", expanded=False):
        st.subheader("Session State")
        st.json(st.session_state.to_dict())
        
        st.subheader("Last Few Messages")
        for i, msg in enumerate(st.session_state.conversation_history[-5:]):
            st.write(f"{i+1}. {msg['role']}: {msg['content'][:100]}...")
        
        # Test buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Test RAG System"):
                try:
                    test_response = rag_system.get_response("Hello")
                    st.success(f"✅ RAG Test Passed: {test_response[:100]}...")
                except Exception as e:
                    st.error(f"❌ RAG Test Failed: {str(e)}")
        
        with col2:
            if st.button("Test Research System"):
                try:
                    test_research = self_research.receive_and_save_research("Test")
                    st.success(f"✅ Research Test Passed: {test_research[:100]}...")
                except Exception as e:
                    st.error(f"❌ Research Test Failed: {str(e)}")

# Export and tools
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("💾 Export Chat", use_container_width=True):
        if st.session_state.conversation_history:
            export_data = {
                'export_date': datetime.now().isoformat(),
                'conversation': st.session_state.conversation_history
            }
            
            # Create download
            st.download_button(
                "Download JSON",
                json.dumps(export_data, indent=2),
                f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
        else:
            st.warning("No chat history to export")

with col2:
    if st.button("🔄 Reset App", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
