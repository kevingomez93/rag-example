import os
from openai import OpenAI
from typing import List, Dict, Any, Optional
from milvus_client import MilvusClient
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 milvus_host: str = "localhost",
                 milvus_port: str = "19530",
                 collection_name: str = "rag_documents"):
        
        # Initialize OpenAI
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize Milvus client
        self.milvus_client = MilvusClient(
            host=milvus_host,
            port=milvus_port,
            collection_name=collection_name
        )
        
        # Connect and setup
        self.setup()
        
    def setup(self):
        """Setup connections and collections"""
        if not self.milvus_client.connect():
            raise RuntimeError("Failed to connect to Milvus")
        
        if not self.milvus_client.create_collection():
            raise RuntimeError("Failed to create/access collection")
        
        logger.info("RAG Engine initialized successfully")
    
    def add_documents(self, documents: List[str], sources: List[str]) -> bool:
        """Add documents to the knowledge base"""
        if len(documents) != len(sources):
            logger.error("Number of documents and sources must match")
            return False
        
        return self.milvus_client.insert_documents(documents, sources)
    
    def retrieve_context(self, query: str, num_docs: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the query"""
        return self.milvus_client.search_similar(query, limit=num_docs)
    
    def generate_response(self, 
                         query: str, 
                         context_docs: List[Dict[str, Any]], 
                         model: str = "gpt-3.5-turbo") -> str:
        """Generate response using retrieved context"""
        
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"Source: {doc['source']}\nContent: {doc['text']}"
            for doc in context_docs
        ])
        
        # Create prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
Use the context below to answer the user's question. If the answer cannot be found in the context, 
say "I don't have enough information to answer that question based on the provided context."

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"Error generating response: {str(e)}"
    
    def chat(self, query: str, num_docs: int = 3, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Main chat function that performs RAG"""
        
        # Retrieve relevant documents
        context_docs = self.retrieve_context(query, num_docs)
        
        # Generate response
        response = self.generate_response(query, context_docs, model)
        
        return {
            "query": query,
            "response": response,
            "context_docs": context_docs,
            "num_retrieved": len(context_docs)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return self.milvus_client.get_collection_stats()


# Utility functions for document processing
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Find the last sentence ending to avoid cutting mid-sentence
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_break = max(last_period, last_newline)
            
            if last_break > start + chunk_size // 2:  # Only break if we're not too close to start
                chunk = chunk[:last_break + 1]
                end = start + len(chunk)
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
            
    return [chunk for chunk in chunks if chunk.strip()]


def process_pdf(file_path: str) -> List[str]:
    """Extract text from PDF and chunk it"""
    try:
        import PyPDF2
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        return chunk_text(text)
    
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")
        return [] 