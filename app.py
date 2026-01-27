import streamlit as st
import time
import json
from datetime import datetime
import pickle
import os
from rag_system import rag_system
from ResearchSystem import self_research

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = {}

# Load saved chats from file on startup
def load_saved_chats():
    try:
        if os.path.exists("saved_chats.pkl"):
            with open("saved_chats.pkl", "rb") as f:
                st.session_state.saved_chats = pickle.load(f)
    except:
        st.session_state.saved_chats = {}

load_saved_chats()

# Save chats to file
def save_chats_to_file():
    try:
        with open("saved_chats.pkl", "wb") as f:
            pickle.dump(st.session_state.saved_chats, f)
    except Exception as e:
        st.error(f"Error saving chats: {e}")

with st.sidebar:
    st.title('RamanTech')
    
    # Color pickers in a bordered container stColumn st-emotion-cache-lzn8yf ek2vi382
    with st.container(border=True):
        st.subheader('Theme')
        
        # Check if data exists in URL
        if "data" in st.query_params:
            # Load from URL
            st.session_state.user_data = json.loads(st.query_params["data"])
            background = st.session_state.user_data['back']
            circle_color = st.session_state.user_data['color']
        else:
            # Set defaults
            background = '#000000'
            circle_color = '#FF0000'

        # Create columns for color pickers
        col1, col2 = st.columns(2)

        with col1:
            background = st.color_picker('Back', value=background)
        with col2:
            circle_color = st.color_picker('Circle', value=circle_color)

        # Save to session state and URL
        st.session_state.user_data = {"back": background, "color": circle_color}
        st.query_params["data"] = json.dumps(st.session_state.user_data)

    # Chat Management Section
    st.divider()
    st.subheader("💬 Chat Management")
    
    # New Chat Button
    if st.button("🆕 New Chat", use_container_width=True, type="primary"):
        # Save current chat if it has messages
        if st.session_state.chat_history:
            st.session_state.saved_chats[st.session_state.current_chat_id] = {
                "name": f"Chat_{len(st.session_state.saved_chats) + 1}",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "messages": st.session_state.chat_history.copy(),
                "background": background,
                "circle_color": circle_color
            }
            save_chats_to_file()
        
        # Start new chat
        st.session_state.chat_history = []
        st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.rerun()
    
    # Save Current Chat Button
    if st.button("💾 Save Current Chat", use_container_width=True):
        if st.session_state.chat_history:
            chat_name = f"Chat_{len(st.session_state.saved_chats) + 1}"
            st.session_state.saved_chats[st.session_state.current_chat_id] = {
                "name": chat_name,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "messages": st.session_state.chat_history.copy(),
                "background": background,
                "circle_color": circle_color
            }
            save_chats_to_file()
            st.success(f"Chat '{chat_name}' saved!")
        else:
            st.warning("No messages to save!")
    
    # Load Saved Chats
    st.divider()
    st.subheader("📂 Saved Chats")
    
    if st.session_state.saved_chats:
        for chat_id, chat_info in st.session_state.saved_chats.items():
            
            with st.container(border=True):
                chat_name = chat_info.get('name', 'Chat')
                chat_date = chat_info.get('date', '')
                if st.button(f"📝 {chat_name}", 
                           key=f"load_{chat_id}", 
                           use_container_width=True):
                    st.session_state.chat_history = chat_info.get("messages", [])
                    st.session_state.current_chat_id = chat_id
                    # Note: Can't automatically set color pickers, but can show info
                    st.info(f"Loaded: {chat_name} ({len(chat_info.get('messages', []))} messages)")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️", key=f"delete_{chat_id}"):
                            if chat_id in st.session_state.saved_chats:
                                del st.session_state.saved_chats[chat_id]
                                save_chats_to_file()
                                st.rerun()
                    with col2:
                        if st.button("📋", key=f"copy_{chat_id}", help="Copy chat ID"):
                            st.code(chat_id)
    else:
        st.write("No saved chats yet")
    
    # Chat Statistics
    st.divider()
    st.subheader("📊 Statistics")
    st.write(f"Current Chat: {len(st.session_state.chat_history)} messages")
    st.write(f"Saved Chats: {len(st.session_state.saved_chats)}")
    
    # Export Chat as JSON
    if st.session_state.chat_history:
        st.divider()
        chat_json = json.dumps({
            "chat_id": st.session_state.current_chat_id,
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.chat_history,
            "background": background,
            "circle_color": circle_color
        }, indent=2)
        
        st.download_button(
            label="📥 Export Chat",
            data=chat_json,
            file_name=f"chat_{st.session_state.current_chat_id}.json",
            mime="application/json",
            use_container_width=True
        )

# Logo
sidebar_logo = 'logo.png'
home_logo = 'logo1.png'
st.logo(sidebar_logo, icon_image=home_logo)

# Clear button
# if st.button("Clear"):
#     st.query_params.clear()
#     st.session_state.user_data = {"name": "", "score": 0}
#     st.rerun()

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

# Custom CSS with dynamic colors
st.markdown(f"""
<style>
    .stLogo  {{
        height: 150px;
    }}
    .ek2vi381:first-child {{
    margin: auto;
    }}
    .e1o8oa9v0{{
        margin-left: -20px;
    }}
    .e1o8oa9v3{{
        margin-left: -30px;
    }}
    .e1q4kxr41{{
    background: linear-gradient(to left, {background}, {circle_color});
    }} 
    .stApp {{
        background: radial-gradient(circle at 130%, {circle_color} 25%, {background} 75%);
        min-height: 100vh;
        padding-bottom: 50px;
    }}
    
    .stMarkdown {{
        z-index: 1;
    }}
     #raman-tech{{
        background: linear-gradient(to right, white, {circle_color});
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        display: inline-block;
    }}
   .e4man113, .st-emotion-cache-1353z0o, .stAppToolbar, .stAppHeader {{
        background: transparent;
         backdrop-filter: blur(10px);
    }}
     /* Style the entire sidebar */
    [data-testid="stSidebar"] {{
        background: radial-gradient(circle at -130%, {circle_color} 25%, {background} 75%);
        padding-top: 0;
        box-shadow: 0 0 30px {circle_color};
    }}

    [data-baseweb="textarea"], [data-baseweb="base-input"] {{
        background: transparent;
        backdrop-filter: blur(10px);
    }}
     
    .stChatInput{{
        background: rgba(255,255,255,0.2);
        border-radius: 50px;
    }}
    
    [data-baseweb="base-input"]:focus {{
        border: none !important;
    }}
    
    /* Remove any error state styling */
    .stChatInput textarea[data-error="true"] {{
        border-color: #ccc !important;
    }}
    
    /* Remove all red outlines */
    .stChatInput textarea {{
        border-color: #ccc !important;
    }}
  
    .eyzqfg11{{
        backdrop-filter: blur(10px);
    }}
    
    /* Style the entire sidebar */
    [data-testid="stSidebar"] {{
        background: radial-gradient(circle at -130%, {circle_color}25%, {background} 75%);
        padding-top: 0;
        box-shadow: 0 0 30px {circle_color};
    }}
    
    .chat-message-container {{
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
    }}
    
    .chat-footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: transperant;
        color: white;
        padding: 10px 20px;
        text-align: center;
        font-size: 0.6em;
        z-index: 1000;
    }}
    .chat-badge {{
        background: {circle_color};
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-left: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# Main content
st.title('RamanTech')
st.write('How can I help you...?')

# Chat badge showing current chat info
if st.session_state.chat_history:
    st.markdown(f'<span class="chat-badge">💬 Chat: {len(st.session_state.chat_history)} messages</span>', 
                unsafe_allow_html=True)

def chat(prompt):
    """Simulate chat response"""
    with st.spinner("Thinking...", show_time=True):
        response = rag_system.get_response(prompt)

    # Check if research needed
    response_lower = str(response).lower()
    needs_research = any(phrase in response_lower for phrase in NO_ANSWER_PHRASES)
    
    if needs_research:
        with st.spinner("Researching additional information..."):
            research_response = self_research.receive_and_save_research(prompt)
            if research_response and str(research_response).strip():
                response = research_response[0] if research_response[1] is None else research_response[0]
                research_data = {
                    'text': f'{prompt}: {research_response}'
                }
                
                with open('research_responses.json', 'w') as f:
                    json.dump(research_data, f, indent=4)

    for char in response:
        yield char
        time.sleep(0.02)

def save_feedback(index):
    """Save user feedback for a message"""
    st.session_state.chat_history[index]["feedback"] = st.session_state.get(f"feedback_{index}")

# Display chat history
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                feedback = message.get("feedback", None)
                st.session_state[f"feedback_{i}"] = feedback
                # Show timestamp if available
                if "timestamp" in message:
                    st.caption(f"Sent at: {message['timestamp']}")
                
                # Show feedback option
                st.feedback(
                    "thumbs",
                    key=f"feedback_{i}",
                    disabled=feedback is not None,
                    on_change=save_feedback,
                    args=[i],
                )

# Chat input
if prompt := st.chat_input("Say something"):
    # Add user message to history
    with st.chat_message("user"):
        st.write(prompt)
    
    st.session_state.chat_history.append({
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    
    with st.chat_message("assistant"):
        response = st.write_stream(chat(prompt))
        # Add feedback option
        st.feedback(
            "thumbs",
            key=f"feedback_{len(st.session_state.chat_history)}",
            on_change=save_feedback,
            args=[len(st.session_state.chat_history)],
        )
    
    # Add assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    
    # Auto-save chat after 5 messages
    if len(st.session_state.chat_history) > 0 and len(st.session_state.chat_history) % 5 == 0:
        if st.session_state.current_chat_id not in st.session_state.saved_chats:
            st.session_state.saved_chats[st.session_state.current_chat_id] = {
                "name": f"AutoSave_{len(st.session_state.saved_chats) + 1}",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "messages": st.session_state.chat_history.copy(),
                "background": background,
                "circle_color": circle_color
            }
            save_chats_to_file()

# Footer
st.markdown(f"""
<div class="chat-footer">
     RamanTech Chat System | Active Chat ID: {st.session_state.current_chat_id} 
</div>
""",
