#!/usr/bin/env python3
"""
Retrieval-Augmented Generation (RAG) System

This script implements a RAG system that uses:
- Milvus as a vector database for storing document embeddings
- Ollama (default) or OpenAI for generating embeddings and responses

The system retrieves relevant documents from Milvus based on query similarity
and uses them as context for generating responses.

Environment variables:
    OPENAI_API_KEY: Your OpenAI API key (optional, only needed when using OpenAI)
"""

import logging
import os
import argparse
import sys
from typing import List, Dict, Any, Optional, Union, Literal

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

# Try to import OpenAI, but don't fail if not available
try:
    import openai
    OPENAI_AVAILABLE = True
    # Initialize with API key from environment variable if available
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not found. Only Ollama will be available.")


class ModelProvider:
    """Enum-like class for model providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"


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

    def initialize_collection(self, vector_dim: int = 1536) -> None:
        """
        Initialize the Milvus collection, creating it if it doesn't exist.

        Args:
            vector_dim: Dimension of embedding vectors (1536 for OpenAI, 4096 for DeepSeek)
        """
        try:
            if not utility.has_collection(self.collection_name):
                self._create_collection(vector_dim)
            else:
                self._load_existing_collection()

            # Load collection into memory
            self.collection.load()
            logger.info(f"Collection '{self.collection_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    def _create_collection(self, vector_dim: int) -> None:
        """
        Create a new Milvus collection with appropriate schema.

        Args:
            vector_dim: Dimension of embedding vectors
        """
        # Define collection fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        ]

        # Create collection with schema
        schema = CollectionSchema(fields, description="Markdown Knowledge Base")
        self.collection = Collection(self.collection_name, schema)
        logger.info(f"Created new collection '{self.collection_name}' with vector dimension {vector_dim}")

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
    """Handles generating embeddings for text using Ollama or OpenAI models."""

    def __init__(
        self,
        provider: str = ModelProvider.OLLAMA,
        model: str = "deepseek-llm",
        openai_model: str = "text-embedding-3-small",
        fallback_to_ollama: bool = True
    ):
        """
        Initialize the embedding service.

        Args:
            provider: The provider to use (ollama or openai)
            model: Name of the Ollama model to use for embeddings
            openai_model: Name of the OpenAI model to use for embeddings
            fallback_to_ollama: Whether to fallback to Ollama if OpenAI fails
        """
        self.provider = provider
        self.model = model
        self.openai_model = openai_model
        self.fallback_to_ollama = fallback_to_ollama

        # Validate OpenAI setup if selected as provider
        if self.provider == ModelProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                if self.fallback_to_ollama:
                    logger.warning("OpenAI package not available, falling back to Ollama")
                    self.provider = ModelProvider.OLLAMA
                else:
                    raise ImportError("OpenAI package is not installed. Please install it with 'pip install openai'")
            elif not openai.api_key:
                if self.fallback_to_ollama:
                    logger.warning("OpenAI API key not found in environment, falling back to Ollama")
                    self.provider = ModelProvider.OLLAMA
                else:
                    raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable")

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the current model.

        Returns:
            Dimension of the embedding vector
        """
        if self.provider == ModelProvider.OPENAI:
            # OpenAI embedding dimensions
            dimensions = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            return dimensions.get(self.openai_model, 1536)
        else:
            # Ollama model dimensions
            dimensions = {
                "deepseek-llm": 4096,
                "llama2": 4096,
                "mistral": 4096,
                "nomic-embed-text": 768
            }
            return dimensions.get(self.model, 4096)

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding vector for the given text.

        Args:
            text: Input text for embedding

        Returns:
            List of float values representing the embedding vector
        """
        if self.provider == ModelProvider.OPENAI:
            try:
                logger.info(f"Generating embedding with OpenAI model: {self.openai_model}")
                response = openai.embeddings.create(
                    model=self.openai_model,
                    input=text,
                    encoding_format="float"
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Error generating OpenAI embedding: {e}")
                if self.fallback_to_ollama:
                    logger.info(f"Falling back to Ollama for embedding")
                    return self._generate_ollama_embedding(text)
                return None
        else:
            return self._generate_ollama_embedding(text)

    def _generate_ollama_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding using Ollama.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            logger.info(f"Generating embedding with Ollama model: {self.model}")
            response = ollama.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error generating Ollama embedding: {e}")
            return None


class LLMService:
    """Handles interaction with Large Language Models via Ollama or OpenAI."""

    def __init__(
        self,
        provider: str = ModelProvider.OLLAMA,
        ollama_model: str = "deepseek-llm",
        openai_model: str = "gpt-3.5-turbo",
        fallback_to_ollama: bool = True
    ):
        """
        Initialize the LLM service.

        Args:
            provider: The provider to use (ollama or openai)
            ollama_model: Name of the Ollama model to use
            openai_model: Name of the OpenAI model to use
            fallback_to_ollama: Whether to fallback to Ollama if OpenAI fails
        """
        self.provider = provider
        self.ollama_model = ollama_model
        self.openai_model = openai_model
        self.fallback_to_ollama = fallback_to_ollama

        # Validate OpenAI setup if selected
        if self.provider == ModelProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                if self.fallback_to_ollama:
                    logger.warning("OpenAI package not available, falling back to Ollama")
                    self.provider = ModelProvider.OLLAMA
                else:
                    raise ImportError("OpenAI package is not installed. Please install it with 'pip install openai'")
            elif not openai.api_key:
                if self.fallback_to_ollama:
                    logger.warning("OpenAI API key not found in environment, falling back to Ollama")
                    self.provider = ModelProvider.OLLAMA
                else:
                    raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable")

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
        if self.provider == ModelProvider.OPENAI:
            try:
                logger.info(f"Generating response with OpenAI model: {self.openai_model}")
                response = openai.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error generating OpenAI response: {e}")
                if self.fallback_to_ollama:
                    logger.info(f"Falling back to Ollama for response generation")
                    return self._generate_ollama_response(messages, temperature)
                return None
        else:
            return self._generate_ollama_response(messages, temperature)

    def _generate_ollama_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float
    ) -> Optional[str]:
        """
        Generate response using Ollama.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        try:
            logger.info(f"Generating response with Ollama model: {self.ollama_model}")
            response = ollama.chat(
                model=self.ollama_model,
                messages=messages,
                options={"temperature": temperature}
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating Ollama response: {e}")
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
        # Get the vector dimension from the selected embedding model
        vector_dim = self.embedding_service.get_embedding_dimension()

        # Initialize Milvus with the correct vector dimension
        self.milvus_manager.connect()
        self.milvus_manager.initialize_collection(vector_dim)

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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrieval-Augmented Generation (RAG) system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--provider", "-p",
        choices=["ollama", "openai"],
        default="ollama",
        help="Model provider to use for embeddings and generation"
    )

    parser.add_argument(
        "--embedding-model", "-em",
        default="deepseek-llm",
        help="Model name for embeddings (for Ollama: deepseek-llm, nomic-embed-text; for OpenAI: text-embedding-3-small)"
    )

    parser.add_argument(
        "--llm-model", "-lm",
        default="deepseek-llm",
        help="Model name for text generation (for Ollama: deepseek-llm, llama2; for OpenAI: gpt-3.5-turbo, gpt-4o)"
    )

    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Temperature for text generation (0.0-1.0)"
    )

    parser.add_argument(
        "--results", "-r",
        type=int,
        default=3,
        help="Number of relevant documents to retrieve"
    )

    parser.add_argument(
        "--milvus-host",
        default="localhost",
        help="Milvus server host"
    )

    parser.add_argument(
        "--milvus-port",
        default="19530",
        help="Milvus server port"
    )

    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback to Ollama if OpenAI fails"
    )

    return parser.parse_args()


def check_openai_availability() -> bool:
    """
    Check if OpenAI is available and configured.

    Returns:
        True if OpenAI is available and API key is set
    """
    if not OPENAI_AVAILABLE:
        print("⚠️ OpenAI package is not installed. To use OpenAI, install with: pip install openai")
        return False

    if not openai.api_key:
        print("⚠️ OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        return False

    return True


def select_openai_models(embedding_model: str, llm_model: str) -> tuple:
    """
    Select appropriate OpenAI models based on user input.

    Args:
        embedding_model: User specified embedding model
        llm_model: User specified LLM model

    Returns:
        Tuple of (embedding_model, llm_model)
    """
    # Default OpenAI models
    default_embed = "text-embedding-3-small"
    default_llm = "gpt-3.5-turbo"

    # Map of common names to OpenAI model names
    embed_map = {
        "ada": "text-embedding-ada-002",
        "small": "text-embedding-3-small",
        "large": "text-embedding-3-large",
    }

    llm_map = {
        "gpt3": "gpt-3.5-turbo",
        "gpt4": "gpt-4o"
    }

    # Resolve embedding model
    final_embed = embed_map.get(embedding_model.lower(), embedding_model)
    if not any(model in final_embed for model in ["text-embedding", "ada"]):
        print(f"⚠️ '{embedding_model}' may not be a valid OpenAI embedding model, using {default_embed}")
        final_embed = default_embed

    # Resolve LLM model
    final_llm = llm_map.get(llm_model.lower(), llm_model)
    if not any(model in final_llm for model in ["gpt-3", "gpt-4"]):
        print(f"⚠️ '{llm_model}' may not be a valid OpenAI LLM model, using {default_llm}")
        final_llm = default_llm

    return final_embed, final_llm


def main() -> int:
    """
    Main entry point for the RAG application.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Parse command line arguments
        args = parse_arguments()

        # If provider is OpenAI, check availability and API key
        if args.provider == ModelProvider.OPENAI and not check_openai_availability():
            if args.no_fallback:
                print("❌ OpenAI not available or not configured, and fallback is disabled")
                return 1
            else:
                print("⚠️ Falling back to Ollama")
                args.provider = ModelProvider.OLLAMA

        # Initialize components
        milvus_manager = MilvusManager(host=args.milvus_host, port=args.milvus_port)

        # Configure models based on provider
        if args.provider == ModelProvider.OPENAI:
            # Select appropriate OpenAI models
            openai_embed_model, openai_llm_model = select_openai_models(
                args.embedding_model, args.llm_model
            )

            # Initialize services with OpenAI models
            embedding_service = EmbeddingService(
                provider=ModelProvider.OPENAI,
                model=args.embedding_model,
                openai_model=openai_embed_model,
                fallback_to_ollama=not args.no_fallback
            )

            llm_service = LLMService(
                provider=ModelProvider.OPENAI,
                ollama_model=args.llm_model,
                openai_model=openai_llm_model,
                fallback_to_ollama=not args.no_fallback
            )

            print(f"🔌 Using OpenAI models: {openai_embed_model} (embedding), {openai_llm_model} (generation)")
        else:
            # Initialize services with Ollama models
            embedding_service = EmbeddingService(
                provider=ModelProvider.OLLAMA,
                model=args.embedding_model
            )

            llm_service = LLMService(
                provider=ModelProvider.OLLAMA,
                ollama_model=args.llm_model
            )

            print(f"🔌 Using Ollama models: {args.embedding_model} (embedding), {args.llm_model} (generation)")

        # Set up RAG system
        rag_system = RAGSystem(milvus_manager, embedding_service, llm_service)
        rag_system.setup()

        # Interactive question answering loop
        print("\n🤖 RAG System ready")
        print("Type 'exit' to quit\n")

        while True:
            query = input("\n🔍 Please enter your question: ")
            if query.lower() in ("exit", "quit", "q"):
                break

            print("\n⏳ Searching and generating answer...")
            response = rag_system.answer_question(
                query,
                result_limit=args.results,
                temperature=args.temperature
            )

            if response:
                print(f"\n🤖 Answer: {response}")
            else:
                print("\n❌ Sorry, I couldn't generate a response.")

        # Clean up resources
        rag_system.cleanup()
        return 0

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error in main program: {e}", exc_info=True)
        print(f"\n❌ An error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())