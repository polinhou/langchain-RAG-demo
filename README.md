# Local Knowledge Base Q&A with Ollama and Qdrant

A question-answering system that uses Ollama's local language models and Qdrant vector database to answer questions based on your PDF documents.

## Features

- Uses Ollama's local models for text embedding and Q&A
- Efficient vector search with Qdrant
- Supports Chinese Q&A
- Simple command-line interface
- Processes multiple PDF files
- Lightweight and easy to set up

## Prerequisites

- Python 3.11
- Poetry (Python package manager)
- Ollama (local language model service)
- Docker and Docker Compose (for Qdrant)

## Quick Start

### 1. Create Required Directories

Create the necessary directories for storing data and Qdrant storage:

```bash
# Create data directory for storing PDF files
mkdir -p data

# Create directory for Qdrant persistent storage
mkdir -p qdrant_storage
```

### 2. Install Ollama

Refer to the [Ollama documentation](https://ollama.ai/) to install and start the Ollama service.

### 3. Download Models

```bash
ollama pull bge-m3:latest  # For text embedding
ollama pull gemma3:latest  # For Q&A
```

### 4. Start Qdrant Service

```bash
docker-compose up -d｀
```

### 5. Install Python Dependencies

```bash
poetry install
```

### 6. Prepare Documents

Place your PDF files in the `data/` directory you created earlier.

### 7. Run the Application

```bash
poetry run python main.py
```

## Usage

1. After starting the application, enter your question
2. The system will answer based on the PDF files in the `data/` directory
3. Type `q` to exit the program

## Environment Variables

You can customize the settings by creating a `.env` file:

```
# Ollama Settings
OLLAMA_EMBEDDINGS_URL=http://localhost:11434
OLLAMA_EMBEDDINGS_MODEL=bge-m3:latest
OLLAMA_CHAT_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=gemma3:latest

# Qdrant Settings
QDRANT_URL=:memory:
COLLECTION_NAME=qa_collection

# File Paths
DIRPATH=./data
```

## License

MIT License

3. Run the demo:
   ```bash
   poetry run python main.py
   ```

## Configuration

You can configure the application using the following environment variables in the `.env` file:

```
# Ollama Configuration
OLLAMA_EMBEDDINGS_URL=http://host.docker.internal:11434  # For Docker
OLLAMA_EMBEDDINGS_MODEL=bge-m3:latest
OLLAMA_CHAT_URL=http://host.docker.internal:11434
OLLAMA_CHAT_MODEL=gemma3:latest

# Qdrant Configuration
QDRANT_URL=http://qdrant:6333
COLLECTION_NAME=qa_collection

# Application
DIRPATH=./data
```

## Docker Compose Services

- `qdrant`: Vector database service
- `app`: The main application service

## Volumes

- `qdrant_storage`: Persistent storage for Qdrant data
- `./data`: Local directory for PDF files (mounted to /app/data in the container)

## Using a Different Model

You can use any embedding model supported by Ollama. For example, to use the `nomic-embed-text` model:

1. Pull the model:
   ```bash
   ollama pull nomic-embed-text
   ```

2. Update your `.env` file:
   ```env
   OLLAMA_MODEL=nomic-embed-text:latest
   ```

3. Run the demo again:
   ```bash
   poetry run python main.py
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=bge-m3:latest

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=lyrics_collection
```

## Quick Start

1. Start Qdrant server using Docker:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. Run the demo:
   ```bash
   poetry run python main.py
   ```

3. The script will:
   - Generate embeddings using Azure OpenAI
   - Store vectors in Qdrant
   - Perform a similarity search
   - Display the most similar results

## Project Structure

```
.
├── main.py                # Main application script
├── pyproject.toml         # Project dependencies and configuration
├── README.md              # This file
├── .env.example           # Example environment variables
└── .gitignore             # Git ignore rules
```

## Development

### Adding New Features

1. Install development dependencies:
   ```bash
   poetry install --with dev
   ```

2. Run tests:
   ```bash
   poetry run pytest
   ```

3. Format code:
   ```bash
   poetry run black .
   ```

## Security

- Never commit your `.env` file or any API keys to version control
- The `.gitignore` is configured to exclude sensitive files
- Use environment variables for all sensitive configuration

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue first to discuss what you would like to change.

## Support

For support, please open an issue or contact the maintainers.
