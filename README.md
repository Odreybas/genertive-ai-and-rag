# genertive-ai-and-rag

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, HuggingFace Transformers, and FAISS for vector search.

## Setup

1. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your PDF file in the workspace directory.

4. Run the script:
   ```bash
   python main.py
   ```

5. Enter the path to your PDF file when prompted.

6. Enter your query when prompted.

The system will process the PDF, create embeddings, and answer questions based on the content using FLAN-T5 model.