import streamlit as st
import re
import torch
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="RAG Q&A", layout="centered", initial_sidebar_state="collapsed")
st.title("📄 RAG Q&A System")
st.markdown("Upload PDFs and ask questions instantly")

# ==============================
# HELPER FUNCTIONS
# ==============================
def fix_pdf_text(text):
    """Clean extracted PDF text"""
    text = re.sub(r'(?<= )([^ ]) (?=[^ ])', r'\1', text)
    text = re.sub(r'^([^ ]) (?=[^ ])', r'\1', text)
    text = re.sub(r'\n([^ ]) ', r'\n\1', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_available_pdfs():
    """Get list of PDFs in pdfs folder"""
    pdf_folder = Path("pdfs")
    pdf_folder.mkdir(exist_ok=True)
    return sorted([f.name for f in pdf_folder.glob("*.pdf")])

# ==============================
# FILE UPLOAD SECTION
# ==============================
st.markdown("### 📤 Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_folder = Path("pdfs")
    pdf_folder.mkdir(exist_ok=True)
    pdf_path = pdf_folder / uploaded_file.name
    
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.success(f"✅ Saved: {uploaded_file.name}")

# ==============================
# PDF SELECTOR
# ==============================
st.markdown("---")
pdf_files = get_available_pdfs()

if not pdf_files:
    st.warning("⚠️ No PDFs found. Upload one above to get started!")
    st.stop()

selected_pdf = st.selectbox("📚 Select PDF to analyze:", pdf_files)

# ==============================
# LOAD RAG PIPELINE (per PDF)
# ==============================
@st.cache_resource
def load_rag_pipeline(pdf_name):
    """Load and cache RAG pipeline for specific PDF"""
    pdf_path = Path("pdfs") / pdf_name
    
    # Load PDF
    with st.spinner("📖 Loading PDF..."):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.page_content = fix_pdf_text(page.page_content)
    
    # Split into chunks
    with st.spinner("✂️ Splitting text..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
    
    # Create embeddings and vector store
    with st.spinner("🔍 Creating embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    
    # Load model (cache once)
    with st.spinner("🤖 Loading LLM (FLAN-T5)..."):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = 'google/flan-t5-large'
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    
    return {
        'retriever': retriever,
        'tokenizer': tokenizer,
        'model': model,
        'device': device,
        'pdf_name': pdf_name
    }

# Load pipeline for selected PDF
pipeline = load_rag_pipeline(selected_pdf)

# ==============================
# QUERY FUNCTION
# ==============================
def query_rag(question, pipeline):
    """Answer question using RAG with improved intelligence"""
    retriever = pipeline['retriever']
    tokenizer = pipeline['tokenizer']
    model = pipeline['model']
    device = pipeline['device']
    
    # Get more relevant documents for better context
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
    
    return answer, context

# ==============================
# Q&A INTERFACE
# ==============================
st.markdown("---")
st.markdown(f"### 💬 Ask Questions About: `{selected_pdf}`")

question = st.text_input("❓ Your question:", placeholder="e.g., What are the main skills?")

if question:
    with st.spinner("⏳ Thinking..."):
        answer, context = query_rag(question, pipeline)
    
    st.markdown("### 📝 Answer")
    st.write(answer)
    
    with st.expander("📌 View Retrieved Context"):
        st.text(context)
else:
    st.info("👆 Ask a question above to get started!")
