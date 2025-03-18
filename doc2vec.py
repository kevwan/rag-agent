#!/usr/bin/env python3
"""
Markdown Vectorizer

This script processes Markdown files into a Milvus vector database for semantic search.
It extracts text from Markdown files, generates embeddings using Ollama with DeepSeek model,
and stores them in a Milvus collection for later retrieval.

Usage:
    python improved_markdown_vectorizer.py --dir /path/to/markdown/files --skip node_modules .git dist
"""

import os
import argparse
import sys
from typing import List, Optional, Dict, Any

import ollama
from markdown_it import MarkdownIt
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)


class MilvusConnection:
    """Manages connection to Milvus database."""

    def __init__(self, host: str = "localhost", port: str = "19530", alias: str = "default"):
        """
        Initialize and establish connection to Milvus.

        Args:
            host: Milvus server host
            port: Milvus server port
            alias: Connection alias
        """
        self.host = host
        self.port = port
        self.alias = alias
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Milvus server."""
        connections.connect(alias=self.alias, host=self.host, port=self.port)
        print(f"✅ Connected to Milvus at {self.host}:{self.port}")

    def disconnect(self) -> None:
        """Close connection to Milvus server."""
        connections.disconnect(self.alias)
        print(f"✅ Disconnected from Milvus at {self.host}:{self.port}")


class MarkdownProcessor:
    """Handles Markdown file scanning and text extraction."""

    def __init__(self, md_parser: MarkdownIt = None):
        """
        Initialize Markdown processor with a parser.

        Args:
            md_parser: Markdown parser instance
        """
        self.md_parser = md_parser or MarkdownIt()

    def extract_text(self, md_file: str) -> str:
        """
        Extract text content from a Markdown file.

        Args:
            md_file: Path to Markdown file

        Returns:
            Rendered text content from the Markdown file
        """
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            return self.md_parser.render(content)
        except Exception as e:
            print(f"❌ Error extracting text from {md_file}: {e}")
            return ""

    def scan_files(self, directory: str, skip_folders: Optional[List[str]] = None) -> List[str]:
        """
        Recursively scan directory for Markdown files.

        Args:
            directory: Root directory to scan
            skip_folders: List of folder names to skip

        Returns:
            List of Markdown file paths
        """
        if skip_folders is None:
            skip_folders = []

        if not os.path.isdir(directory):
            print(f"❌ Error: {directory} is not a valid directory")
            return []

        md_files = []
        try:
            for root, dirs, files in os.walk(directory, topdown=True):
                # Skip unwanted folders
                dirs[:] = [d for d in dirs if d not in skip_folders]

                for file in files:
                    if file.endswith(".md"):
                        md_files.append(os.path.join(root, file))
        except Exception as e:
            print(f"❌ Error scanning directory {directory}: {e}")

        print(f"📝 Found {len(md_files)} Markdown files in {directory}")
        return md_files


class EmbeddingGenerator:
    """Handles text embedding generation."""

    def __init__(self, model: str = "deepseek-llm"):
        """
        Initialize embedding generator with specified model.

        Args:
            model: Name of the Ollama model to use
        """
        self.model = model

    def generate(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Input text to generate embedding for

        Returns:
            Embedding vector as list of floats
        """
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            raise


class MilvusCollection:
    """Manages Milvus collection operations."""

    def __init__(
        self,
        collection_name: str = "markdown_knowledge_base",
        vector_dim: int = 4096,
        max_content_length: int = 65535
    ):
        """
        Initialize Milvus collection manager.

        Args:
            collection_name: Name of the Milvus collection
            vector_dim: Dimension of embedding vectors
            max_content_length: Maximum length for content field
        """
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.max_content_length = max_content_length
        self.collection = self._setup_collection()

    def _setup_collection(self) -> Collection:
        """
        Set up Milvus collection with appropriate schema.

        Returns:
            Initialized Milvus collection
        """
        # Define collection fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=self.max_content_length),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]

        # Drop collection if it exists
        if utility.has_collection(self.collection_name):
            print(f"⚠️ Collection '{self.collection_name}' already exists, dropping...")
            utility.drop_collection(self.collection_name)

        # Create new collection
        schema = CollectionSchema(fields, description="Markdown Knowledge Base")
        collection = Collection(self.collection_name, schema)
        print(f"✅ Created collection '{self.collection_name}'")
        return collection

    def insert_document(self, filename: str, content: str, embedding: List[float]) -> None:
        """
        Insert document into Milvus collection.

        Args:
            filename: Path of the original file
            content: Text content
            embedding: Vector embedding
        """
        # Truncate content if necessary
        if len(content) > self.max_content_length:
            print(f"⚠️ Content length {len(content)} truncated to {self.max_content_length}")
            content = content[:self.max_content_length]

        # Insert data
        self.collection.insert([
            [filename],    # List of filenames
            [content],     # List of contents
            [embedding]    # List of embeddings
        ])

    def create_index(self, index_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Create index on the embedding field.

        Args:
            index_params: Optional custom index parameters
        """
        if index_params is None:
            index_params = {
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {
                    "M": 8,
                    "efConstruction": 200
                }
            }

        self.collection.create_index("embedding", index_params)
        print(f"✅ Created index on 'embedding' field")

    def load(self) -> None:
        """Load collection into memory for searching."""
        self.collection.load()
        print(f"✅ Loaded collection '{self.collection_name}' into memory")

    def flush(self) -> None:
        """Flush collection data to disk."""
        self.collection.flush()


class MarkdownVectorizer:
    """Main class for processing Markdown documents into vector database."""

    def __init__(
        self,
        milvus_connection: MilvusConnection,
        markdown_processor: MarkdownProcessor,
        embedding_generator: EmbeddingGenerator,
        milvus_collection: MilvusCollection
    ):
        """
        Initialize vectorizer with component instances.

        Args:
            milvus_connection: MilvusConnection instance
            markdown_processor: MarkdownProcessor instance
            embedding_generator: EmbeddingGenerator instance
            milvus_collection: MilvusCollection instance
        """
        self.milvus_connection = milvus_connection
        self.markdown_processor = markdown_processor
        self.embedding_generator = embedding_generator
        self.milvus_collection = milvus_collection

    def process_documents(self, directory: str, skip_folders: Optional[List[str]] = None) -> None:
        """
        Process all Markdown documents in directory.

        Args:
            directory: Root directory containing Markdown files
            skip_folders: List of folder names to skip
        """
        # Get all Markdown files
        md_files = self.markdown_processor.scan_files(directory, skip_folders)
        total_files = len(md_files)
        processed = 0

        if total_files == 0:
            print(f"⚠️ No Markdown files found in {directory}")
            return

        # Process each file
        for md_file in md_files:
            try:
                # Extract text from Markdown
                text = self.markdown_processor.extract_text(md_file)
                if not text:
                    continue

                # Generate embedding
                embedding = self.embedding_generator.generate(text)

                # Insert to Milvus
                self.milvus_collection.insert_document(md_file, text, embedding)

                # Update progress
                processed += 1
                print(f"📥 Processed {processed}/{total_files}: {md_file}")

                # Flush periodically
                if processed % 10 == 0:
                    self.milvus_collection.flush()

            except Exception as e:
                print(f"❌ Failed to process {md_file}: {e}")

        # Final flush
        self.milvus_collection.flush()
        print(f"✅ Processed {processed}/{total_files} Markdown files")

    def create_search_index(self) -> None:
        """Create search index and load collection."""
        self.milvus_collection.create_index()
        self.milvus_collection.load()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process Markdown files into Milvus vector database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dir", "-d",
        dest="markdown_dir",
        required=True,
        help="Directory containing Markdown files to process"
    )

    parser.add_argument(
        "--skip", "-s",
        nargs="+",
        default=["node_modules", ".git", "backup"],
        help="List of directories to skip during processing"
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="Milvus server host"
    )

    parser.add_argument(
        "--port",
        default="19530",
        help="Milvus server port"
    )

    parser.add_argument(
        "--model",
        default="deepseek-llm",
        help="Ollama model name for embeddings"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the script.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse command line arguments
    args = parse_arguments()

    # Expand user directory if present (e.g., ~/documents)
    markdown_dir = os.path.expanduser(args.markdown_dir)

    # Validate directory
    if not os.path.isdir(markdown_dir):
        print(f"❌ Error: {markdown_dir} is not a valid directory")
        return 1

    # Initialize components
    try:
        milvus_conn = MilvusConnection(host=args.host, port=args.port)
        md_processor = MarkdownProcessor()
        embedding_gen = EmbeddingGenerator(model=args.model)
        milvus_coll = MilvusCollection()

        # Create vectorizer
        vectorizer = MarkdownVectorizer(
            milvus_connection=milvus_conn,
            markdown_processor=md_processor,
            embedding_generator=embedding_gen,
            milvus_collection=milvus_coll
        )

        # Process documents
        print(f"🔍 Processing Markdown files from: {markdown_dir}")
        print(f"🚫 Skipping directories: {args.skip}")

        vectorizer.process_documents(markdown_dir, args.skip)
        vectorizer.create_search_index()
        print("✅ Vectorization complete!")

        # Disconnect from Milvus
        milvus_conn.disconnect()
        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())