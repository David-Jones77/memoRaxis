# GraphDB (Neo4j) Replacement for ChromaDB

This document provides instructions for replacing ChromaDB with GraphDB (Neo4j) in the memoRaxis system.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Implementation Details](#implementation-details)
4. [Configuration](#configuration)
5. [Installation](#installation)
6. [Testing](#testing)
7. [Usage](#usage)
8. [Troubleshooting](#troubleshooting)

## Overview

The memoRaxis system has been modified to use Neo4j as a GraphDB replacement for ChromaDB. This change provides several benefits:

- Better support for complex relationships between memories
- More flexible querying capabilities
- Improved scalability for large datasets
- More natural representation of memory links

## Prerequisites

Before using the GraphDB replacement, you need to:

1. **Install Neo4j**: Download and install Neo4j from [https://neo4j.com/download/](https://neo4j.com/download/).
2. **Start Neo4j**: Start the Neo4j database service.
3. **Enable Vector Index**: Ensure that Neo4j 5.12+ is installed, as vector index functionality is required.
4. **Create a Neo4j database**: Create a new database or use an existing one.
5. **Set Neo4j credentials**: Ensure you have the correct username and password for Neo4j.

## Implementation Details

### Files Modified

1. **A-mem/retrievers.py**: Removed ChromaRetriever import
2. **A-mem/memory_system.py**: Modified to use Neo4jRetriever instead of ChromaRetriever
3. **requirements.txt**: Added Neo4j Python driver dependency

### Files Created

1. **A-mem/graphdb_retriever.py**: Implements Neo4jRetriever class
2. **test_graphdb_replacement.py**: Test script for GraphDB replacement

### Key Components

#### Neo4jRetriever Class

The `Neo4jRetriever` class implements the same interface as `ChromaRetriever`, but uses Neo4j for storage and retrieval:

- **Initialization**: Creates a Neo4j driver and vector index
- **add_document**: Adds a memory document to Neo4j with vector embedding
- **delete_document**: Deletes a memory document from Neo4j
- **search**: Searches for similar documents using vector similarity
- **reset**: Resets the Neo4j database by deleting all memory nodes

#### Vector Search Implementation

The Neo4jRetriever uses Neo4j's vector index feature to perform similarity search:

1. **Vector Embedding**: Uses the same SentenceTransformer model as ChromaDB
2. **Vector Index**: Creates a vector index for efficient similarity search
3. **Cosine Similarity**: Uses cosine similarity to find similar documents

## Configuration

### Neo4j Connection Settings

By default, the Neo4jRetriever connects to Neo4j at `bolt://localhost:7687` with username `neo4j` and password `password`.

To change these settings, modify the `__init__` method of the `Neo4jRetriever` class in `A-mem/graphdb_retriever.py`:

```python
def __init__(self, collection_name: str = "memories", model_name: str = "all-MiniLM-L6-v2",
             uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
```

### Vector Index Configuration

The vector index is configured with the following settings:

- **Dimensions**: 384 (matches the all-MiniLM-L6-v2 model)
- **Similarity Function**: cosine

To change these settings, modify the `_create_vector_index` method in `A-mem/graphdb_retriever.py`:

```python
session.run("""
    CREATE VECTOR INDEX memory_vector_index
    IF NOT EXISTS
    FOR (m:Memory)
    ON (m.embedding)
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: 384,
            `vector.similarity_function`: 'cosine'
        }
    }
""")
```

## Installation

### 1. Install Neo4j

Download and install Neo4j from [https://neo4j.com/download/](https://neo4j.com/download/).

### 2. Start Neo4j

Start the Neo4j database service:

- **Windows**: Start the Neo4j service from the Services console
- **macOS**: Run `neo4j start` in the terminal
- **Linux**: Run `sudo systemctl start neo4j` or `neo4j start`

### 3. Install Python Dependencies

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Testing

### Run Test Script

Run the test script to verify the GraphDB replacement:

```bash
python test_graphdb_replacement.py
```

### Test Results

The test script will test the following functionality:

1. **Initialization**: Tests if AMemMemorySystem can be initialized
2. **Add Memory**: Tests if memories can be added to the system
3. **Retrieve Memory**: Tests if memories can be retrieved by similarity
4. **Reset**: Tests if the memory system can be reset
5. **Add After Reset**: Tests if memories can be added after reset

## Usage

### Basic Usage

```python
from src.amem_memory import AMemMemorySystem

# Initialize memory system
memory = AMemMemorySystem()

# Add memory
memory.add_memory("Python is a popular programming language", {"tags": ["programming", "language"], "context": "Technology"})

# Retrieve memory
results = memory.retrieve("programming language", top_k=2)
for result in results:
    print(result.content)

# Reset memory system
memory.reset()
```

### Advanced Usage

#### Custom Neo4j Settings

To use custom Neo4j settings, modify the `Neo4jRetriever` initialization in `A-mem/graphdb_retriever.py`:

```python
# Example: Custom Neo4j settings
def __init__(self, collection_name: str = "memories", model_name: str = "all-MiniLM-L6-v2",
             uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "mysecretpassword"):
```

#### Custom Embedding Model

To use a different embedding model, modify the `model_name` parameter:

```python
# Example: Use a different embedding model
memory = AMemMemorySystem(model_name="paraphrase-multilingual-MiniLM-L12-v2")
```

## Troubleshooting

### Common Issues

#### 1. Neo4j Connection Error

**Error**: `Failed to initialize AMemMemorySystem: Connection refused`

**Solution**: Ensure Neo4j is running and accessible at the configured URI.

#### 2. Neo4j Authentication Error

**Error**: `Failed to initialize AMemMemorySystem: Auth Failed`

**Solution**: Check your Neo4j username and password in `graphdb_retriever.py`.

#### 3. Neo4j Vector Index Error

**Error**: `Failed to initialize AMemMemorySystem: No such procedure 'gds.similarity.cosine'`

**Solution**: Ensure Neo4j 5.12+ is installed and the GDS library is enabled.

#### 4. Neo4j Driver Error

**Error**: `ModuleNotFoundError: No module named 'neo4j'`

**Solution**: Install the Neo4j Python driver: `pip install neo4j`

### Debugging Tips

1. **Check Neo4j Status**: Ensure Neo4j is running and accessible
2. **Verify Neo4j Credentials**: Check your username and password
3. **Check Neo4j Version**: Ensure Neo4j 5.12+ is installed
4. **Enable Neo4j Logs**: Check Neo4j logs for more detailed error information
5. **Test Neo4j Connection**: Use the Neo4j Browser to test connection and queries

## Conclusion

The GraphDB (Neo4j) replacement for ChromaDB provides a more flexible and powerful storage solution for the memoRaxis system. By using Neo4j, the system can better handle complex relationships between memories and provide more flexible querying capabilities.

With the provided implementation and instructions, you can easily switch from ChromaDB to Neo4j in your memoRaxis system.