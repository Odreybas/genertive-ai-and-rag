# ==============================
# SECTION 1: INSTALL LIBRARIES
# ==============================

# Note: Run `pip install -r requirements.txt` to install dependencies

# ==============================
# SECTION 2: IMPORTS
# ==============================

import re
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================
# SECTION 3: TEXT CLEANING FUNCTION
# ==============================

def fix_pdf_text(text):
    text = re.sub(r'(?<= )([^ ]) (?=[^ ])', r'\1', text)
    text = re.sub(r'^([^ ]) (?=[^ ])', r'\1', text)
    text = re.sub(r'\n([^ ]) ', r'\n\1', text)
    return re.sub(r'\s+', ' ', text).strip()

# ==============================
# SECTION 4: LOAD PDF
# ==============================

import os

raw_path = input('Please enter the path to your PDF file (local workspace path is recommended): ')
pdf_path = raw_path.strip().strip('"').strip("'")

if not pdf_path:
    raise SystemExit('No file path provided. Exiting.')

if not os.path.isabs(pdf_path):
    pdf_path = os.path.join(os.getcwd(), pdf_path)

if not os.path.isfile(pdf_path):
    raise FileNotFoundError(f"File path {pdf_path!r} is not a valid file. Make sure the file exists in the container workspace.")

loader = PyPDFLoader(pdf_path)
pages = loader.load()

for page in pages:
    page.page_content = fix_pdf_text(page.page_content)

# ==============================
# SECTION 5: SPLIT INTO CHUNKS
# ==============================

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    print(f"Total chunks created: {len(chunks)}")

# ==============================
# SECTION 6: EMBEDDINGS AND VECTOR STORE
# ==============================

    print('Creating embeddings and vector store...')

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# ==============================
# SECTION 7: LOAD THE LLM (FLAN-T5)
# ==============================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = 'google/flan-t5-large'

    print(f'Loading {model_id} on {device}...')

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

# ==============================
# SECTION 8: RAG QUERY FUNCTION
# ==============================

    def query_rag(question):
        relevant_docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Enhanced prompt for better answers
        input_text = f"""Based on the following context, provide a comprehensive and complete answer to the question.

Context: {context}

Question: {question}

Answer:"""

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # Improved generation parameters for longer, more complete answers
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,  # Increased from 512
            min_new_tokens=50,    # Ensure minimum length
            do_sample=True,       # Enable sampling for more natural text
            temperature=0.3,      # Low temperature for coherence
            top_p=0.9,           # Nucleus sampling
            repetition_penalty=1.2,  # Reduce repetition
            length_penalty=1.0,   # Neutral length preference
            num_beams=4,         # Beam search for better quality
            early_stopping=True
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to ensure complete sentences
        answer = answer.strip()
        
        # If answer ends mid-sentence, try to complete it
        if answer and not answer.endswith(('.', '!', '?', ':')):
            # Find last complete sentence
            sentences = answer.split('. ')
            if len(sentences) > 1:
                # Keep all complete sentences
                complete_sentences = []
                for i, sent in enumerate(sentences):
                    if i < len(sentences) - 1:  # All but last
                        complete_sentences.append(sent + '.')
                    else:
                        # Check if last part is a complete sentence
                        if sent.strip() and sent.strip()[-1] in '.!?':
                            complete_sentences.append(sent)
                        # Otherwise, discard incomplete sentence
                
                if complete_sentences:
                    answer = ' '.join(complete_sentences)
        
        return answer

# ==============================
# SECTION 9: RUN A QUERY
# ==============================

    print(f'\n--- RAG Output for: {pdf_path} ---')

    user_query = input("Enter your query: ")

    print(query_rag(user_query))

else:
    print('No file path provided.')