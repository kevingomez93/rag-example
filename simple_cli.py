#!/usr/bin/env python3
"""
Simple CLI for RAG Chat with Milvus
A command-line interface for testing the RAG functionality
"""

import os
import sys
from rag_engine import RAGEngine, chunk_text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("🤖 RAG Chat with Milvus - CLI Version")
    print("=" * 50)
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        openai_key = input("Enter your OpenAI API Key: ").strip()
        if not openai_key:
            print("❌ OpenAI API key is required!")
            sys.exit(1)
    
    # Milvus configuration
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    collection_name = os.getenv("COLLECTION_NAME", "rag_documents")
    
    print(f"Connecting to Milvus at {milvus_host}:{milvus_port}")
    print(f"Using collection: {collection_name}")
    
    try:
        # Initialize RAG engine
        rag_engine = RAGEngine(
            openai_api_key=openai_key,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            collection_name=collection_name
        )
        print("✅ RAG Engine initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG Engine: {e}")
        sys.exit(1)
    
    # Add sample documents
    print("\n📚 Adding sample documents...")
    sample_docs = [
        """
        Artificial Intelligence (AI) is a broad field of computer science focused on creating systems 
        capable of performing tasks that typically require human intelligence. These tasks include 
        learning, reasoning, problem-solving, perception, and language understanding. AI has applications 
        in various domains including healthcare, finance, transportation, and entertainment.
        """,
        """
        Machine Learning is a subset of artificial intelligence that focuses on algorithms that can 
        learn from and make predictions or decisions based on data. Unlike traditional programming 
        where rules are explicitly coded, machine learning systems improve their performance through 
        experience and exposure to data.
        """,
        """
        Vector databases are specialized databases designed to store and search high-dimensional vectors 
        efficiently. They are particularly useful for AI applications involving embeddings, such as 
        semantic search, recommendation systems, and similarity matching. Milvus is a popular open-source 
        vector database that supports various indexing algorithms and similarity metrics.
        """,
        """
        Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with 
        text generation. It works by first retrieving relevant documents from a knowledge base using 
        semantic search, then using those documents as context for a language model to generate 
        accurate and informed responses. This approach helps reduce hallucinations in AI responses.
        """
    ]
    
    sources = [
        "AI Overview",
        "Machine Learning Basics", 
        "Vector Databases Guide",
        "RAG Explanation"
    ]
    
    if rag_engine.add_documents(sample_docs, sources):
        print(f"✅ Added {len(sample_docs)} sample documents!")
    else:
        print("❌ Failed to add sample documents")
    
    # Get stats
    stats = rag_engine.get_stats()
    print(f"📊 Knowledge base contains {stats.get('num_entities', 0)} documents")
    
    print("\n💬 Chat Interface")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'stats' to see knowledge base statistics")
    print("-" * 50)
    
    # Chat loop
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                stats = rag_engine.get_stats()
                print(f"📊 Knowledge base statistics:")
                print(f"   - Documents: {stats.get('num_entities', 0)}")
                print(f"   - Collection: {stats.get('name', 'N/A')}")
                continue
            
            if not user_input:
                print("Please enter a question or 'quit' to exit.")
                continue
            
            print("🤖 Assistant: ", end="", flush=True)
            
            # Get response
            result = rag_engine.chat(user_input)
            print(result["response"])
            
            # Show sources if available
            if result["context_docs"]:
                print(f"\n📚 Sources ({len(result['context_docs'])} documents):")
                for i, doc in enumerate(result["context_docs"], 1):
                    print(f"   {i}. {doc['source']} (score: {doc['score']:.3f})")
                    print(f"      {doc['text'][:100]}...")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 