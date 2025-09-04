import os
import sys
import types
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class DummyCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="stubbed")
            )]
        )


class DummyChat:
    def __init__(self):
        self.completions = DummyCompletions()


class DummyOpenAI:
    def __init__(self, api_key=None):
        self.chat = DummyChat()


class DummyMilvusClient:
    def __init__(self, host="localhost", port="19530", collection_name="rag_documents"):
        self.host = host
        self.port = port
        self.collection_name = collection_name

    def connect(self):
        return True

    def create_collection(self):
        return True

    def insert_documents(self, texts, sources):
        self.inserted = (texts, sources)
        return True

    def search_similar(self, query, limit=5):
        self.last_query = query
        self.last_limit = limit
        return []

    def get_collection_stats(self):
        return {}


# Stub external modules before importing rag_engine
pymilvus_stub = types.ModuleType("pymilvus")
pymilvus_stub.connections = types.SimpleNamespace(connect=lambda *a, **k: None)
pymilvus_stub.Collection = object
pymilvus_stub.FieldSchema = object
pymilvus_stub.CollectionSchema = object
pymilvus_stub.DataType = object
pymilvus_stub.utility = types.SimpleNamespace(has_collection=lambda name: False)
sys.modules.setdefault("pymilvus", pymilvus_stub)

sentence_stub = types.ModuleType("sentence_transformers")
sentence_stub.SentenceTransformer = lambda *a, **k: None
sys.modules.setdefault("sentence_transformers", sentence_stub)

openai_stub = types.ModuleType("openai")
openai_stub.OpenAI = DummyOpenAI
sys.modules.setdefault("openai", openai_stub)

import rag_engine


@pytest.fixture
def rag(monkeypatch):
    monkeypatch.setattr(rag_engine, "MilvusClient", DummyMilvusClient)
    monkeypatch.setattr(rag_engine, "OpenAI", DummyOpenAI)
    return rag_engine
