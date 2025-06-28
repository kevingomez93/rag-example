# 🤖 RAG Chat with Milvus - Simple POC

A complete Retrieval-Augmented Generation (RAG) chat system using Python and the Milvus vector database.

---

## 🌟 Features

* **Vector-based document retrieval** with Milvus
* **Semantic search** using sentence transformers
* **LLM integration** with OpenAI GPT models
* **Streamlit** web interface
* **Command-line interface** for testing
* **PDF document support**
* **Text chunking** with overlap for better context
* **Source attribution** for responses

---

## 🚀 Quick Start Guide (macOS with Colima)

### Prerequisites

* Python 3.8+ (tested with Python 3.13)
* OpenAI API key ([get one here](https://platform.openai.com/account/api-keys))
* Colima with Docker (recommended on macOS)

---

### 1️⃣ Install Colima (only once)

```bash
brew install colima
```

### 2️⃣ Start Colima

```bash
colima start
```

(You **do not** need `--with-docker`; Docker is default in Colima.)

---

### 3️⃣ Clone and Setup Environment

```bash
git clone https://github.com/kevingomez93/rag-example.git
cd rag-example
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 4️⃣ Configure API Key

```bash
export OPENAI_API_KEY="your-openai-key-here"
```

👉 **Tip:** Add this permanently to your shell profile:

```bash
echo 'export OPENAI_API_KEY="your-openai-key-here"' >> ~/.zshrc
source ~/.zshrc
```

---

### 5️⃣ Start Milvus with Docker (inside Colima)

```bash
docker-compose up -d
```

Check the containers:

```bash
docker ps
```

---

## 🖥️ Running the Application

### Web Interface (Streamlit)

```bash
source venv/bin/activate
export OPENAI_API_KEY="your-openai-key-here"
streamlit run streamlit_app.py
```

**Open your browser at:** [http://localhost:8501](http://localhost:8501)

#### Using the Streamlit Web UI:

1. Click **Add Sample Documents** or upload your own
2. Start asking questions about your documents!

---

### Command Line Interface

```bash
source venv/bin/activate
export OPENAI_API_KEY="your-openai-key-here"
python simple_cli.py
```

#### CLI Commands:

* Type questions to chat with your documents
* Type `stats` to see knowledge base statistics
* Type `quit` or `exit` to end the session

---

## 📊 Example Usage

```plaintext
🧑 You: What is RAG?

🤖 Assistant: Retrieval-Augmented Generation (RAG) combines document retrieval with generative models to produce informed answers, by retrieving relevant text from a knowledge base and feeding it into an LLM.

📚 Sources:
   1. RAG Explanation (score: 0.433)
```

---

## 🛠️ Managing Milvus (with Colima/Docker)

* **Start:**

  ```bash
  docker-compose up -d
  ```
* **Stop:**

  ```bash
  docker-compose down
  ```
* **Stop and remove volumes:**

  ```bash
  docker-compose down -v
  ```
* **Check status:**

  ```bash
  docker ps
  ```

---

## 🐛 Troubleshooting

| Problem                                 | Solution                                                 |
| --------------------------------------- | -------------------------------------------------------- |
| `command not found: streamlit`          | Check venv is activated, reinstall requirements          |
| `OpenAI API key error`                  | Ensure `OPENAI_API_KEY` is exported                      |
| `Failed to connect to Milvus`           | Make sure Colima is running and Docker containers are up |
| `huggingface/tokenizers` warnings       | Set `export TOKENIZERS_PARALLELISM=false`                |
| `externally-managed-environment` errors | Always use a Python venv                                 |

**Debug steps:**

* Confirm: `which python` shows your `venv/bin/python`
* Check Milvus logs: `docker-compose logs`
* Confirm API key: `echo $OPENAI_API_KEY`
* Check installed dependencies:

  ```bash
  pip list | grep -E "(streamlit|pymilvus|openai)"
  ```

---

## 📁 Project Structure

```
rag-example/
├── milvus_client.py
├── rag_engine.py
├── streamlit_app.py
├── simple_cli.py
├── requirements.txt
├── docker-compose.yml
├── env_example.txt
└── README.md
```

---

## ⚙️ Configuration

| Variable          | Description            | Default        | Required |
| ----------------- | ---------------------- | -------------- | -------- |
| `OPENAI_API_KEY`  | Your OpenAI API key    | None           | ✅ Yes    |
| `MILVUS_HOST`     | Milvus server host     | localhost      | No       |
| `MILVUS_PORT`     | Milvus port            | 19530          | No       |
| `COLLECTION_NAME` | Milvus collection name | rag\_documents | No       |

**Embedding Model**: `all-MiniLM-L6-v2` (384 dims)
**LLM**: `gpt-3.5-turbo` (or `gpt-4`)
**Similarity**: Cosine
**Index type**: IVF\_FLAT

---

## 🔧 Advanced Usage

### Add Your Own Documents (via code)

```python
from rag_engine import RAGEngine

rag = RAGEngine()

docs = ["your document text", "another document"]
sources = ["Doc 1", "Doc 2"]
rag.add_documents(docs, sources)

result = rag.chat("Your question")
print(result["response"])
```

---

## 🚦 Production Considerations

✅ Use external Milvus cluster
✅ Add proper rate-limiting and authentication
✅ Add logging and monitoring
✅ Cache embeddings for speed
✅ Validate user inputs

---

## 🆘 Getting Help

* Check the console logs
* Validate your environment variables
* Confirm Milvus is running
* Make sure your OpenAI key has credits
* Use:

  ```bash
  docker-compose logs
  ```

  for deeper debugging.

---

## 📞 Quick Commands

```bash
# First-time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Every time you work
source venv/bin/activate
export OPENAI_API_KEY="your-key"

# Start Milvus with Colima
colima start
docker-compose up -d

# Run web
streamlit run streamlit_app.py

# Run CLI
python simple_cli.py

# Tear down
docker-compose down -v
colima stop
```
