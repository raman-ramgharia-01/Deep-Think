import os
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

class RAGSystem:
    def __init__(self):
        # Initialize model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load data
        try:
            self.df = joblib.load('normalize_data.joblib')
            print(f"DataFrame loaded. Shape: {self.df.shape}")
        except FileNotFoundError:
            print("Error: 'normalize_data.joblib' not found")
            self.df = None
    
    def analyze_with_groq(self, text_data):
        """Send text to Groq API and get response"""
        try:
            client = Groq(
                api_key= os.environ.get("GROQ_API_KEY")
            )

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
            print(f"Error calling Groq API: {e}")
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
        rag_prompt = f"""
        [Role]
        You are responding as Raman — the user. Use first-person ("I", "my", "me") naturally.
        
        [Context]
        {retrieved_context}
        
        [Question from Rohan]
        {user_query}
        
        [Rules]
        - Answer **strictly based on the given context**.
        - Write as if Rohan himself is speaking.
        - If the context doesn't have enough info, say: "I don't have enough information to answer this based on what I know."
        - Keep the tone personal, friendly, and genuine.
        
        [Rohan's Answer]
        """
        
        # Save prompt (optional)
        with open("prompt.txt", 'w', encoding='utf-8') as f:
            f.write(rag_prompt)
        
        # Get response from Groq
        response = self.analyze_with_groq(rag_prompt)
        
        return response

# Initialize RAG system
rag_system = RAGSystem()
