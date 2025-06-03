import os
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusClient:
    def __init__(self, host: str = "localhost", port: str = "19530", collection_name: str = "rag_documents"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2 model
        
    def connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False
    
    def create_collection(self):
        """Create collection if it doesn't exist"""
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            
            schema = CollectionSchema(fields, "RAG documents collection")
            self.collection = Collection(self.collection_name, schema)
            
            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index("embedding", index_params)
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings"""
        return self.encoder.encode(texts)
    
    def insert_documents(self, texts: List[str], sources: List[str]) -> bool:
        """Insert documents into the collection"""
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return False
            
            # Generate embeddings
            embeddings = self.encode_text(texts)
            
            # Prepare data
            data = [
                texts,
                sources,
                embeddings.tolist()
            ]
            
            # Insert data
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Inserted {len(texts)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            return False
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return []
            
            # Load collection
            self.collection.load()
            
            # Encode query
            query_embedding = self.encode_text([query])
            
            # Search parameters
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            # Perform search
            results = self.collection.search(
                query_embedding.tolist(),
                "embedding",
                search_params,
                limit=limit,
                output_fields=["text", "source"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "text": hit.entity.get("text"),
                        "source": hit.entity.get("source"),
                        "score": hit.score
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}
            
            self.collection.load()
            stats = {
                "name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "schema": str(self.collection.schema)
            }
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)} 