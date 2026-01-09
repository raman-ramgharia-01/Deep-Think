import streamlit as st
import json
import time
from datetime import datetime
from rag_system import rag_system
from ResearchSystem import self_research

# Set page config
st.set_page_config(
    page_title="Chat Assistant",
    page_icon="💬",
    layout="wide"
)

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'clear_chat' not in st.session_state:
    st.session_state.clear_chat = False

# Initialize no-answer phrases
NO_ANSWER_PHRASES = [
    "don't have enough information",
    "no information provided",
    "cannot answer",
    "not enough context",
    "i don't know",
    "i don't have"
]

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #4a90e2;
    }
    .assistant-message {
        background-color: #e8f4fd;
        border-left: 4px solid #00c853;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("💬 Chat Settings")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()
    
    st.markdown("---")
    
    # Display stats
    st.subheader("📊 Chat Stats")
    st.write(f"Total messages: {len(st.session_state.conversation_history)}")
    
    if st.session_state.conversation_history:
        last_message = st.session_state.conversation_history[-1]
        last_time = last_message.get('timestamp', 'N/A')
        st.write(f"Last message: {last_time}")
    
    st.markdown("---")
    
    # About section
    st.subheader("ℹ️ About")
    st.markdown("""
    This is an intelligent chat assistant powered by:
    - RAG (Retrieval-Augmented Generation) system
    - Self-research capabilities
    - Real-time information retrieval
    """)

# Main chat interface
st.title("🤖 Intelligent Chat Assistant")
st.markdown("---")

# Chat container
chat_container = st.container()

# Display chat history
with chat_container:
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(f"**You:** {message['content']}")
                if 'timestamp' in message:
                    st.caption(f"*{message['timestamp']}*")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:** {message['content']}")
                if 'timestamp' in message:
                    st.caption(f"*{message['timestamp']}*")

# Chat input
if prompt := st.chat_input("Type your message here..."):
    if prompt.strip():
        # Add user message to history
        user_message = {
            'role': 'user',
            'content': prompt,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.conversation_history.append(user_message)
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f"**You:** {prompt}")
            st.caption(f"*{user_message['timestamp']}*")
        
        # Create a placeholder for assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Show typing indicator
            with st.spinner("Thinking..."):
                try:
                    # Get initial response from RAG system
                    response = rag_system.get_response(prompt)
                    
                    # Check if answer is insufficient
                    if any(phrase in response.lower() for phrase in NO_ANSWER_PHRASES):
                        response_placeholder.markdown("🔍 **I need to research this further...**")
                        
                        # Get research-based answer
                        research_response = self_research.receive_and_save_research(prompt)
                        
                        # Update response with research
                        response = research_response
                        
                        # Add research indicator
                        st.success("✅ New information added to knowledge base")
                    
                    # Display response in chunks for streaming effect
                    full_response = ""
                    for chunk in response.split():
                        full_response += chunk + " "
                        response_placeholder.markdown(f"**Assistant:** {full_response}")
                        time.sleep(0.05)  # Simulate typing
                    
                    # Add assistant message to history
                    assistant_message = {
                        'role': 'assistant',
                        'content': response,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    }
                    st.session_state.conversation_history.append(assistant_message)
                    
                    # Show timestamp
                    st.caption(f"*{assistant_message['timestamp']}*")
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    response_placeholder.markdown(f"**Assistant:** {error_msg}")
                    
                    # Add error to history
                    error_message = {
                        'role': 'assistant',
                        'content': error_msg,
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'error': True
                    }
                    st.session_state.conversation_history.append(error_message)

# Export functionality
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Export Chat History (JSON)", use_container_width=True):
        if st.session_state.conversation_history:
            export_data = {
                'export_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_messages': len(st.session_state.conversation_history),
                'conversation': st.session_state.conversation_history
            }
            
            # Convert to JSON string
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            # Create download button
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.warning("No chat history to export")

with col2:
    if st.button("📋 Copy Last Response", use_container_width=True):
        if st.session_state.conversation_history:
            last_assistant_msg = None
            for msg in reversed(st.session_state.conversation_history):
                if msg['role'] == 'assistant':
                    last_assistant_msg = msg['content']
                    break
            
            if last_assistant_msg:
                st.code(last_assistant_msg)
                st.success("Response copied to clipboard!")
            else:
                st.warning("No assistant responses found")
        else:
            st.warning("No chat history")

with col3:
    if st.button("🔄 Refresh Chat", use_container_width=True):
        st.rerun()

# Information panel at the bottom
with st.expander("ℹ️ How to use this chat assistant"):
    st.markdown("""
    ### Features:
    1. **Smart Responses**: Uses RAG system for accurate, context-aware answers
    2. **Self-Research**: Automatically researches topics when information is insufficient
    3. **Persistent Chat**: Conversation history is maintained during your session
    4. **Export Options**: Download your chat history as JSON
    
    ### Tips:
    - Ask specific questions for better answers
    - The system will automatically research topics it doesn't know
    - Use the sidebar to manage chat settings
    - Clear chat history when starting a new topic
    
    ### Technical Notes:
    - Maximum 50 messages stored in memory (per session)
    - Research system activates when RAG doesn't have sufficient information
    - All responses are generated in real-time
    """)
