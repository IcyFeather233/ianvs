
## RAG (Retrieval Augmented Generation) Support

This project now includes RAG capabilities powered by LangChain, allowing the LLM to leverage external knowledge during inference.

### What is RAG?

RAG (Retrieval Augmented Generation) is a technique that enhances Large Language Models by:
1. Retrieving relevant information from a knowledge base
2. Incorporating this information into the context when generating responses
3. Helping the model provide more accurate and knowledge-grounded answers

### Features

- **Automatic Knowledge Base Management**
  - Automatic initialization and loading of vector store
  - Support for incremental updates to the knowledge base
  - Document change detection to avoid unnecessary reprocessing
  - Persistent storage of embeddings

- **Vector Store**
  - Uses Chroma as the vector store backend
  - Efficient similarity search for relevant context
  - Automatic embedding generation using Hugging Face models

### Setup and Usage

1. **Configure Knowledge Base**
   
   Edit the following constants in `rag/testalgorithms/gen/basemodel.py`:
   ```python
   KNOWLEDGE_BASE_PATH = "/path/to/your/knowledge/base"  # Directory containing .txt documents
   VECTOR_STORE_PATH = "/path/to/vector/store"          # Where to store embeddings
   EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model to use
   ```

2. **Prepare Documents**
   - Place your knowledge base documents in the `KNOWLEDGE_BASE_PATH` directory
   - Supported format: `.txt` files
   - Documents will be automatically processed and embedded

3. **Run the Benchmark**
   ```bash
   ianvs -f examples/government/singletask_learning_bench/rag/benchmarkingjob.yaml
   ```
   The system will automatically:
   - Initialize or load the vector store
   - Process any new or modified documents
   - Use RAG during inference

### How It Works

1. **Document Processing**
   - Documents are split into chunks using CharacterTextSplitter
   - Each chunk is embedded using the specified embedding model
   - Embeddings are stored in a Chroma vector store

2. **During Inference**
   - The user query is used to retrieve relevant context
   - Retrieved context is added to the system message
   - The LLM generates a response considering both the query and context

3. **Optimization**
   - Document hashing ensures efficient updates
   - Vector store persistence reduces initialization time
   - Error handling ensures robust operation

### Dependencies

The RAG implementation relies on the following libraries:
- LangChain: For RAG pipeline implementation
- Chroma: For vector store functionality
- Sentence Transformers: For document embedding
- HuggingFace Transformers: For LLM integration

### Best Practices

1. **Document Organization**
   - Keep documents focused and well-organized
   - Use clear, descriptive filenames
   - Regular text files work best

2. **Performance Optimization**
   - Monitor vector store size
   - Consider chunk size based on your use case
   - Adjust the number of retrieved contexts (k) as needed

3. **Maintenance**
   - Regularly update knowledge base content
   - Monitor log output for any issues
   - Back up vector store periodically

### Troubleshooting

Common issues and solutions:
1. **No documents found**: Ensure documents are in .txt format in KNOWLEDGE_BASE_PATH
2. **Embedding errors**: Check EMBEDDING_MODEL availability and internet connection
3. **Memory issues**: Adjust chunk_size and chunk_overlap in text splitter

### Limitations

- Currently only supports .txt files
- Vector store must fit in memory
- Embedding process may take time for large document collections

For more information about RAG and LangChain:
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [RAG Overview](https://python.langchain.com/docs/use_cases/question_answering/)