import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, vector_store, llm_interface):
        self.vector_store = vector_store
        self.llm = llm_interface

    def answer_query(self, query, embedder, k=3):
        """
        End-to-end RAG: Query -> Embed -> Retrieve -> Generate
        """
        # 1. Embed query
        query_embedding = embedder.generate_embeddings([query])[0]
        
        # 2. Retrieve
        _, _, retrieved_chunks = self.vector_store.search(query_embedding, k=k)
        
        context = "\n\n".join([chunk for chunk in retrieved_chunks if chunk])
        
        # 3. Generate
        answer = self.llm.generate(query, context)
        
        return answer, retrieved_chunks

class LLMInterface:
    def generate(self, query, context):
        raise NotImplementedError

class GeminiLLM(LLMInterface):
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None

    def generate(self, query, context):
        if not self.model:
            return "Error: Gemini API Key not provided."
        
        prompt = f"""
        You are a helpful assistant analyzing product reviews.
        Use the following context (customer reviews) to answer the user's question.
        If the answer is not in the context, say "I don't have enough information from the reviews."
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini: {e}"

class MockLLM(LLMInterface):
    """For testing without API key."""
    def generate(self, query, context):
        return f"This is a mock answer based on {len(context.split())} characters of context. The reviews mention positive sentiments mostly."
