#!/usr/bin/env python3
"""
Retrieval-Augmented Generation (RAG) System

This script implements a RAG system that uses:
- Milvus as a vector database for storing document embeddings
- Ollama with DeepSeek-LLM for generating embeddings and responses

The system retrieves relevant documents from Milvus based on query similarity
and uses them as context for generating responses with the DeepSeek LLM.
"""

import logging
from typing import List, Dict, Any, Optional, Union

import ollama
from pymilvus import (
    Collection,
    connections,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_system")


class MilvusManager:
    """Handles connections and operations with the Milvus vector database."""

    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        alias: str = "default"
    ):
        """
        Initialize the Milvus connection manager.

        Args:
            host: Milvus server hostname
            port: Milvus server port
            alias: Connection alias name
        """
        self.host = host
        self.port = port
        self.alias = alias
        self.collection = None
        self.collection_name = "markdown_knowledge_base"

    def connect(self) -> None:
        """Establish connection to Milvus server."""
        try:
            connections.connect(alias=self.alias, host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def initialize_collection(self) -> None:
        """Initialize the Milvus collection, creating it if it doesn't exist."""
        try:
            if not utility.has_collection(self.collection_name):
                self._create_collection()
            else:
                self._load_existing_collection()

            # Load collection into memory
            self.collection.load()
            logger.info(f"Collection '{self.collection_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    def _create_collection(self) -> None:
        """Create a new Milvus collection with appropriate schema."""
        # Define collection fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4096),
        ]

        # Create collection with schema
        schema = CollectionSchema(fields, description="Markdown Knowledge Base")
        self.collection = Collection(self.collection_name, schema)
        logger.info(f"Created new collection '{self.collection_name}'")

        # Create index for vector search
        self._create_index()

    def _load_existing_collection(self) -> None:
        """Load an existing Milvus collection."""
        self.collection = Collection(self.collection_name)
        logger.info(f"Using existing collection '{self.collection_name}'")

    def _create_index(self) -> None:
        """Create an index on the embedding field."""
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
        logger.info(f"Created index on 'embedding' field")

    def is_collection_loaded(self) -> bool:
        """
        Check if the collection is loaded in memory.

        Returns:
            bool: True if the collection exists and is loaded
        """
        try:
            return utility.has_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Error checking if collection is loaded: {e}")
            return False

    def search(
        self,
        vector: List[float],
        limit: int = 3,
        output_fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.

        Args:
            vector: Query embedding vector
            limit: Maximum number of results to return
            output_fields: Collection fields to include in results

        Returns:
            List of matching documents with their metadata
        """
        if output_fields is None:
            output_fields = ["filename", "content"]

        # Ensure collection is loaded
        if not self.is_collection_loaded():
            self.collection.load()

        search_params = {
            "metric_type": "L2",
            "params": {"ef": 100},
        }

        try:
            results = self.collection.search(
                data=[vector],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=output_fields,
            )

            if not results or len(results) == 0:
                logger.warning("No matching documents found")
                return []

            # Format results for easier consumption
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "score": hit.score,
                        "filename": hit.entity.filename,
                        "content": hit.entity.content,
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def disconnect(self) -> None:
        """Close connection to Milvus server."""
        try:
            connections.disconnect(self.alias)
            logger.info(f"Disconnected from Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")


class EmbeddingService:
    """Handles generating embeddings for text using Ollama models."""

    def __init__(self, model: str = "deepseek-llm"):
        """
        Initialize the embedding service.

        Args:
            model: Name of the Ollama model to use for embeddings
        """
        self.model = model

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding vector for the given text.

        Args:
            text: Input text for embedding

        Returns:
            List of float values representing the embedding vector
        """
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None


class LLMService:
    """Handles interaction with Large Language Models via Ollama."""

    def __init__(self, model: str = "deepseek-llm"):
        """
        Initialize the LLM service.

        Args:
            model: Name of the Ollama model to use
        """
        self.model = model

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (higher = more creative)

        Returns:
            The generated text response
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature}
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return None


class RAGSystem:
    """
    Retrieval-Augmented Generation system that combines vector search
    with language model generation.
    """

    def __init__(
        self,
        milvus_manager: MilvusManager,
        embedding_service: EmbeddingService,
        llm_service: LLMService
    ):
        """
        Initialize the RAG system with required components.

        Args:
            milvus_manager: Manager for Milvus operations
            embedding_service: Service for generating embeddings
            llm_service: Service for language model operations
        """
        self.milvus_manager = milvus_manager
        self.embedding_service = embedding_service
        self.llm_service = llm_service

    def setup(self) -> None:
        """Set up the RAG system by initializing all components."""
        self.milvus_manager.connect()
        self.milvus_manager.initialize_collection()

    def search_knowledge_base(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for documents relevant to the query.

        Args:
            query: The search query text
            limit: Maximum number of results to return

        Returns:
            List of relevant documents with metadata
        """
        # Generate embedding for the query
        query_embedding = self.embedding_service.generate_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate embedding for query")
            return []

        # Search Milvus for similar documents
        results = self.milvus_manager.search(
            vector=query_embedding,
            limit=limit,
            output_fields=["filename", "content"]
        )

        return results

    def answer_question(
        self,
        query: str,
        result_limit: int = 3,
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Answer a question using retrieved knowledge and LLM.

        Args:
            query: The user's question
            result_limit: Maximum number of documents to retrieve
            temperature: LLM temperature parameter

        Returns:
            Generated answer text
        """
        # Retrieve relevant documents
        retrieved_docs = self.search_knowledge_base(query, limit=result_limit)
        if not retrieved_docs:
            logger.warning("No relevant documents found for query")
            return self.llm_service.generate_response([
                {"role": "user", "content": f"Please answer this question to the best of your ability: {query}"}
            ], temperature)

        # Extract content from retrieved documents
        context_texts = [doc["content"] for doc in retrieved_docs]
        context = "\n\n".join(context_texts)

        # Create prompt with retrieved context
        prompt = f"""
You are a knowledgeable AI assistant. Please answer the following question
based on the provided Markdown documents:

REFERENCE DOCUMENTS:
{context}

USER QUESTION: {query}

Provide a comprehensive and accurate answer based on the reference documents.
If the documents don't contain relevant information, state that you don't have
enough information to answer accurately.
"""

        logger.info(f"Generated prompt with {len(retrieved_docs)} context documents")

        # Get response from LLM
        answer = self.llm_service.generate_response([
            {"role": "user", "content": prompt}
        ], temperature)

        return answer

    def cleanup(self) -> None:
        """Clean up resources used by the RAG system."""
        self.milvus_manager.disconnect()


def main() -> None:
    """Main entry point for the RAG application."""
    try:
        # Initialize components
        milvus_manager = MilvusManager()
        embedding_service = EmbeddingService()
        llm_service = LLMService()

        # Set up RAG system
        rag_system = RAGSystem(milvus_manager, embedding_service, llm_service)
        rag_system.setup()

        # Interactive question answering loop
        print("\n🤖 RAG System with DeepSeek-LLM and Milvus")
        print("Type 'exit' to quit\n")

        while True:
            query = input("\n🔍 Please enter your question: ")
            if query.lower() in ("exit", "quit", "q"):
                break

            print("\n⏳ Searching and generating answer...")
            response = rag_system.answer_question(query)

            if response:
                print(f"\n🤖 Answer: {response}")
            else:
                print("\n❌ Sorry, I couldn't generate a response.")

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        logger.error(f"Error in main program: {e}", exc_info=True)
        print(f"\n❌ An error occurred: {e}")
    finally:
        # Clean up resources
        try:
            rag_system.cleanup()
        except Exception:
            pass
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()