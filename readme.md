# Markdown RAG System

A Retrieval Augmented Generation (RAG) system for searching and querying Markdown documents using vector embeddings. The system uses Milvus for vector storage and supports both local Ollama and OpenAI for embeddings and text generation.

## System Overview

This system consists of two main components:

1. **Markdown Vectorizer**: Processes Markdown files, generates embeddings, and stores them in Milvus.
2. **RAG Query System**: Retrieves relevant documents and generates answers to questions using LLMs.

## Requirements

- Python 3.8+
- Docker and Docker Compose (for running Milvus)
- Ollama (for local LLM processing)
- OpenAI API key (optional, for using OpenAI models)

## Setup Instructions

### 1. Start Milvus with Docker Compose

Download the latest Milvus docker-compose.yml file using wget:

```bash
mkdir -p milvus-data
cd milvus-data
wget https://github.com/milvus-io/milvus/releases/download/v2.5.6/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

Start Milvus:

```bash
docker-compose up -d
```

Verify that Milvus is running:

```bash
docker-compose ps
```

You should see containers running for Milvus standalone, Etcd, and MinIO.

### 2. Install Required Python Packages

```bash
pip install pymilvus ollama openai markdown-it
```

### 3. Install and Start Ollama

Download Ollama from [https://ollama.com/](https://ollama.com/) or install it using:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama service:

```bash
ollama serve
```

Pull the required models:

```bash
ollama pull deepseek-llm
```

### 4. Configure Environment (Optional for OpenAI)

If you want to use OpenAI models, set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

For permanent storage, add this line to your `.bashrc` or `.zshrc` file.

### 5. Index Your Markdown Files

Use the Markdown Vectorizer to process your documents:

```bash
python doc2vec.py --dir /path/to/your/markdown/files --skip node_modules .git dist
```

Options:
- `--dir` or `-d`: Directory containing Markdown files
- `--skip` or `-s`: Directories to skip (space-separated)
- `--host`: Milvus host (default: localhost)
- `--port`: Milvus port (default: 19530)
- `--model`: Ollama model for embeddings (default: deepseek-llm)

### 6. Query Your Knowledge Base

Use the RAG system to ask questions about your indexed documents:

```bash
python ragagent.py
```

Options:
- `--provider` or `-p`: Model provider (`ollama` or `openai`, default: ollama)
- `--embedding-model` or `-em`: Model for embeddings
- `--llm-model` or `-lm`: Model for text generation
- `--temperature` or `-t`: Temperature for generation (0.0-1.0, default: 0.7)
- `--results` or `-r`: Number of documents to retrieve (default: 3)
- `--milvus-host`: Milvus server host (default: localhost)
- `--milvus-port`: Milvus server port (default: 19530)
- `--no-fallback`: Disable fallback to Ollama if OpenAI fails

## Examples

### Index a Documentation Repository

```bash
python doc2vec.py --dir ~/projects/documentation --skip node_modules .git assets
```

### Query with Ollama

```bash
python ragagent.py
```

### Query with OpenAI

```bash
# Make sure OPENAI_API_KEY is set
python ragagent.py --provider openai --embedding-model text-embedding-3-small --llm-model gpt-4o
```

## Troubleshooting

1. **Milvus Connection Issues**
   - Ensure Milvus containers are running: `docker ps`
   - Check Milvus logs: `docker logs milvus-standalone`
   - Verify Milvus port is accessible: `curl -I http://localhost:19530`

2. **Ollama Model Issues**
   - Verify Ollama is running: `ps aux | grep ollama`
   - Check available models: `ollama list`
   - Retry pulling models with: `ollama pull model-name`

3. **OpenAI API Issues**
   - Verify your API key is correctly set: `echo $OPENAI_API_KEY`
   - Check for API rate limits or quota issues

4. **Vector Dimension Mismatch**
   - If you encounter errors about vector dimensions, ensure you're using the same embedding model for indexing and querying
   - If necessary, drop the collection and re-index with the desired model

## Maintenance

- **Updating the Knowledge Base**: Re-run the vectorizer script when your Markdown files change
- **Changing Models**: Ensure consistent embedding dimensions or recreate the collection
- **Updating Milvus**: To update to the latest Milvus version:
  ```bash
  cd milvus-data
  docker-compose down
  wget https://github.com/milvus-io/milvus/releases/latest/download/milvus-standalone-docker-compose.yml -O docker-compose.yml
  docker-compose up -d
  ```

## Advanced Configuration

### Custom Milvus Collection Name

To use a different collection name, modify the `collection_name` parameter in the `MilvusCollection` or `MilvusManager` classes.

### Adjusting Vector Dimensions

Different embedding models produce vectors of different dimensions:
- OpenAI text-embedding-3-small: 1536 dimensions
- OpenAI text-embedding-3-large: 3072 dimensions
- DeepSeek and most Ollama models: 4096 dimensions

The system will automatically detect the dimension based on the selected model.