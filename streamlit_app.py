import streamlit as st
import os
from rag_engine import RAGEngine, chunk_text, process_pdf
from typing import List
import tempfile

# Configure page
st.set_page_config(
    page_title="RAG Chat with Milvus",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_added' not in st.session_state:
    st.session_state.documents_added = 0

def init_rag_engine():
    """Initialize RAG engine with user credentials"""
    try:
        rag_engine = RAGEngine(
            openai_api_key=st.session_state.openai_key,
            milvus_host=st.session_state.milvus_host,
            milvus_port=st.session_state.milvus_port,
            collection_name=st.session_state.collection_name
        )
        st.session_state.rag_engine = rag_engine
        st.success("✅ RAG Engine initialized successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG Engine: {str(e)}")
        return False

def add_sample_documents():
    """Add some sample documents for testing"""
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
    
    if st.session_state.rag_engine.add_documents(sample_docs, sources):
        st.session_state.documents_added += len(sample_docs)
        st.success(f"✅ Added {len(sample_docs)} sample documents!")
        return True
    else:
        st.error("❌ Failed to add sample documents")
        return False

# Main App
st.title("🤖 RAG Chat with Milvus")
st.markdown("A simple proof of concept for Retrieval-Augmented Generation using Python and Milvus")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # OpenAI API Key
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key"
    )
    st.session_state.openai_key = openai_key
    
    # Milvus Configuration
    st.subheader("Milvus Settings")
    milvus_host = st.text_input("Milvus Host", value="localhost")
    milvus_port = st.text_input("Milvus Port", value="19530")
    collection_name = st.text_input("Collection Name", value="rag_documents")
    
    st.session_state.milvus_host = milvus_host
    st.session_state.milvus_port = milvus_port
    st.session_state.collection_name = collection_name
    
    # Initialize button
    if st.button("🚀 Initialize RAG Engine", type="primary"):
        if openai_key:
            init_rag_engine()
        else:
            st.error("Please provide OpenAI API key")
    
    # Status
    if st.session_state.rag_engine:
        st.success("✅ RAG Engine Ready")
        
        # Stats
        try:
            stats = st.session_state.rag_engine.get_stats()
            st.metric("Documents in KB", stats.get("num_entities", 0))
        except:
            pass
    else:
        st.warning("⚠️ RAG Engine not initialized")

# Main content area
if st.session_state.rag_engine:
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Add Documents", "📊 Statistics"])
    
    with tab1:
        st.header("Chat with your Knowledge Base")
        
        # Chat interface
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources"):
                        for i, doc in enumerate(message["sources"]):
                            st.write(f"**Source {i+1}:** {doc['source']}")
                            st.write(f"**Score:** {doc['score']:.3f}")
                            st.write(f"**Content:** {doc['text'][:200]}...")
                            st.divider()
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.rag_engine.chat(prompt)
                    
                st.write(result["response"])
                
                # Show sources
                if result["context_docs"]:
                    with st.expander("📚 Sources"):
                        for i, doc in enumerate(result["context_docs"]):
                            st.write(f"**Source {i+1}:** {doc['source']}")
                            st.write(f"**Score:** {doc['score']:.3f}")
                            st.write(f"**Content:** {doc['text'][:200]}...")
                            st.divider()
            
            # Add assistant response
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": result["response"],
                "sources": result["context_docs"]
            })
    
    with tab2:
        st.header("Add Documents to Knowledge Base")
        
        # Sample documents
        if st.button("📝 Add Sample Documents"):
            add_sample_documents()
        
        st.divider()
        
        # Text input
        st.subheader("Add Text Document")
        doc_title = st.text_input("Document Title")
        doc_content = st.text_area("Document Content", height=200)
        
        if st.button("➕ Add Text Document"):
            if doc_title and doc_content:
                chunks = chunk_text(doc_content)
                sources = [f"{doc_title} - Chunk {i+1}" for i in range(len(chunks))]
                
                if st.session_state.rag_engine.add_documents(chunks, sources):
                    st.session_state.documents_added += len(chunks)
                    st.success(f"✅ Added document '{doc_title}' ({len(chunks)} chunks)")
                else:
                    st.error("❌ Failed to add document")
            else:
                st.error("Please provide both title and content")
        
        st.divider()
        
        # File upload
        st.subheader("Upload PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("📤 Process PDF"):
                with st.spinner("Processing PDF..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Process PDF
                        chunks = process_pdf(tmp_file_path)
                        
                        if chunks:
                            sources = [f"{uploaded_file.name} - Chunk {i+1}" for i in range(len(chunks))]
                            
                            if st.session_state.rag_engine.add_documents(chunks, sources):
                                st.session_state.documents_added += len(chunks)
                                st.success(f"✅ Processed '{uploaded_file.name}' ({len(chunks)} chunks)")
                            else:
                                st.error("❌ Failed to add PDF content")
                        else:
                            st.error("❌ Failed to extract text from PDF")
                    
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
    
    with tab3:
        st.header("Knowledge Base Statistics")
        
        if st.button("🔄 Refresh Stats"):
            stats = st.session_state.rag_engine.get_stats()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Documents", stats.get("num_entities", 0))
                st.metric("Documents Added This Session", st.session_state.documents_added)
            
            with col2:
                st.metric("Collection Name", stats.get("name", "N/A"))
            
            if "schema" in stats:
                st.subheader("Collection Schema")
                st.code(stats["schema"])

else:
    # Show setup instructions
    st.info("""
    ## 🚀 Getting Started
    
    1. **Setup Milvus**: Make sure you have Milvus running locally or provide remote connection details
    2. **OpenAI API Key**: Enter your OpenAI API key in the sidebar
    3. **Initialize**: Click the "Initialize RAG Engine" button
    4. **Add Documents**: Use the "Add Documents" tab to populate your knowledge base
    5. **Chat**: Start asking questions about your documents!
    
    ### 🐳 Quick Milvus Setup with Docker
    ```bash
    # Download and run Milvus
    wget https://github.com/milvus-io/milvus/releases/download/v2.3.4/milvus-standalone-docker-compose.yml -O docker-compose.yml
    docker-compose up -d
    ```
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Python, Streamlit, Milvus, and OpenAI</p>
</div>
""", unsafe_allow_html=True) 