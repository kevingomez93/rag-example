# 🤖 RAG Chat with Milvus - Simple POC

A complete Retrieval-Augmented Generation (RAG) chat system using Python and Milvus vector database.

## 🌟 Features

- **Vector-based document retrieval** using Milvus database
- **Semantic search** with sentence transformers
- **LLM integration** with OpenAI GPT models
- **Web interface** built with Streamlit
- **Command-line interface** for testing
- **PDF document processing** support
- **Text chunking** with overlap for better context
- **Source attribution** for generated responses

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.8+** (tested with Python 3.13)
- **OpenAI API key** ([Get one here](https://platform.openai.com/account/api-keys))
- **Docker** (optional - for full Milvus setup)

### Step 1: Clone and Setup Environment

```bash
# Navigate to your project directory
cd new-rag

# Create virtual environment (REQUIRED - don't skip this!)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Key

```bash
# Set your OpenAI API key (replace with your actual key)
export OPENAI_API_KEY="your-openai-api-key-here"
```

**💡 Tip:** Add this to your `~/.zshrc` or `~/.bashrc` to make it permanent:
```bash
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.zshrc
```

### Step 3: Choose Your Setup

You can run this system in two ways:

#### Option A: Embedded Milvus (Recommended for Testing)
Uses Milvus Lite - no Docker required!

#### Option B: Full Milvus with Docker
For production-like setup with persistent storage.

```bash
# Start Milvus containers
docker-compose up -d

# Check if containers are running
docker ps
```

## 🖥️ Running the Application

### Web Interface (Streamlit)

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Set API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Run Streamlit app
streamlit run streamlit_app.py
```

**Open your browser to:** `http://localhost:8501`

#### Using the Web Interface:
1. **Initialize**: The app will auto-detect your API key and connect to Milvus
2. **Add Documents**: Click "Add Sample Documents" or upload your own
3. **Chat**: Start asking questions about your documents!

### Command Line Interface

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Set API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Run CLI
python simple_cli.py
```

#### CLI Commands:
- Type questions to chat with your documents
- Type `stats` to see knowledge base statistics
- Type `quit` or `exit` to end the session

## 📊 Example Usage

### Sample Questions to Try:
- `"What is RAG?"`
- `"How do vector databases work?"`
- `"Explain machine learning"`
- `"What is artificial intelligence?"`

### Expected Output:
```
🧑 You: What is RAG?

🤖 Assistant: Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It works by first retrieving relevant documents from a knowledge base using semantic search, then using those documents as context for a language model to generate accurate and informed responses.

📚 Sources (1 documents):
   1. RAG Explanation (score: 0.433)
```

## 🛠️ Managing Milvus Containers

### Start Milvus:
```bash
docker-compose up -d
```

### Stop Milvus:
```bash
docker-compose down
```

### Stop and Remove All Data:
```bash
docker-compose down -v
```

### Check Status:
```bash
docker ps
```

## 🐛 Troubleshooting

### Common Issues and Solutions:

#### 1. "command not found: streamlit"
**Solution:** Make sure virtual environment is activated:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. "OpenAI API key error"
**Solution:** Check your API key is set correctly:
```bash
echo $OPENAI_API_KEY
export OPENAI_API_KEY="your-actual-key-here"
```

#### 3. "Failed to connect to Milvus"
**Solutions:**
- **For Embedded Mode:** Should work automatically
- **For Docker Mode:** Check containers are running:
  ```bash
  docker-compose ps
  docker-compose up -d
  ```

#### 4. "externally-managed-environment" Error
**Solution:** Always use virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 5. Torch/Tokenizer Warnings
These warnings are harmless and don't affect functionality. To suppress them:
```bash
export TOKENIZERS_PARALLELISM=false
```

## 📁 Project Structure

```
new-rag/
├── milvus_client.py     # Vector database operations
├── rag_engine.py        # Core RAG functionality  
├── streamlit_app.py     # Web interface
├── simple_cli.py        # Command line interface
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Milvus Docker setup
├── env_example.txt      # Environment template
└── README.md           # This file
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | None | ✅ Yes |
| `MILVUS_HOST` | Milvus server hostname | localhost | No |
| `MILVUS_PORT` | Milvus server port | 19530 | No |
| `COLLECTION_NAME` | Milvus collection name | rag_documents | No |

### Model Configuration

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **LLM Model**: `gpt-3.5-turbo` (configurable to `gpt-4`)
- **Similarity Metric**: Cosine similarity
- **Index Type**: IVF_FLAT

## 🔧 Advanced Usage

### Adding Your Own Documents

#### Via Web Interface:
1. Go to "Add Documents" tab
2. Either paste text or upload PDF files
3. Documents are automatically chunked and indexed

#### Via Python Code:
```python
from rag_engine import RAGEngine

# Initialize
rag = RAGEngine()

# Add documents
documents = ["Your document text here...", "Another document..."]
sources = ["Doc 1", "Doc 2"]
rag.add_documents(documents, sources)

# Chat
result = rag.chat("Your question here")
print(result["response"])
```

### Customizing Models

#### Use GPT-4:
```python
result = rag.chat("Your question", model="gpt-4")
```

#### Use Different Embedding Model:
Edit `milvus_client.py`:
```python
self.encoder = SentenceTransformer('all-mpnet-base-v2')
self.dimension = 768  # Update accordingly
```

## 🚦 Production Considerations

For production deployment:

1. **Use external Milvus cluster** for scalability
2. **Implement authentication** and rate limiting
3. **Add proper logging** and monitoring
4. **Use environment variables** for all configuration
5. **Implement caching** for embeddings
6. **Add input validation** and sanitization

## 🆘 Getting Help

### Debug Steps:
1. **Check virtual environment:** `which python` should show path in `venv/`
2. **Check API key:** `echo $OPENAI_API_KEY`
3. **Check Milvus:** `docker ps` or check logs with `docker-compose logs`
4. **Check dependencies:** `pip list | grep -E "(streamlit|pymilvus|openai)"`

### Still Having Issues?
- Check the console output for specific error messages
- Ensure all dependencies are properly installed
- Verify your OpenAI API key has sufficient credits
- Make sure no firewall is blocking connections

## 📞 Quick Command Reference

```bash
# Setup (one time)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Every time you use it
source venv/bin/activate
export OPENAI_API_KEY="your-key"

# Run web interface
streamlit run streamlit_app.py

# Run CLI
python simple_cli.py

# Manage Docker (optional)
docker-compose up -d    # Start
docker-compose down     # Stop
docker ps              # Check status
```

---

🎉 **You're ready to go!** Start with the CLI for quick testing, then try the web interface for a full experience. 