from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from rag_system import rag_system
import json
from ResearchSystem import self_research

app = Flask(__name__)
CORS(app)

# Store conversation history
conversation_history = []

@app.route('/')
def home():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        user_query = data.get('message', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get response from RAG system
        response = rag_system.get_response(user_query)
        
        # if answer not in llm context then 

        no_answer_phrases = [
            "don't have enough information",
            "no information provided",
            "cannot answer",
            "not enough context"
        ]

        if any(phrase in response.lower() for phrase in no_answer_phrases):
            gen_answer = self_research.receive_and_save_research(user_query)

            print('-----* Add New Answer *-----')
            print(f'new answer are added in json or csv')

        # Store in conversation history
        conversation_history.append({
            'user': user_query,
            'assistant': response
        })
        
        # Limit history to last 50 messages
        if len(conversation_history) > 50:
            conversation_history.pop(0)
        
        return jsonify({
            'response': response,
            'history': conversation_history
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': 'An error occurred while processing your request.',
            'response': 'Sorry, I encountered an error. Please try again.'
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    return jsonify({'history': conversation_history})

@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return jsonify({'success': True, 'message': 'History cleared'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)