import streamlit as st
import re
import torch
import json
import os
import base64
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2AudioLM
import pyttsx3
import speech_recognition as sr
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from gtts import gTTS
import tempfile
import base64

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="RAG Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

st.markdown("""
<style>
    .main { max-width: 1400px; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { padding: 12px 24px; }
</style>
""", unsafe_allow_html=True)

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

def save_chat_history(pdf_name, chat):
    """Save chat history to JSON"""
    history_dir = Path("chat_history")
    history_dir.mkdir(exist_ok=True)
    history_file = history_dir / f"{pdf_name.replace('.pdf', '')}_history.json"
    with open(history_file, "w") as f:
        json.dump(chat, f, indent=2)

def load_chat_history(pdf_name):
    """Load chat history from JSON"""
    history_dir = Path("chat_history")
    history_file = history_dir / f"{pdf_name.replace('.pdf', '')}_history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            return json.load(f)
    return []

def speak_text(text):
    """Convert text to speech using gTTS and provide audio player"""
    try:
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
        
        # Generate speech
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_filename)
        
        # Read file and encode as base64 for Streamlit audio player
        with open(temp_filename, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        # Clean up temp file
        os.unlink(temp_filename)
        
        # Create audio player in Streamlit
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio controls autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        
        return True
    except Exception as e:
        st.error(f"❌ Voice synthesis failed: {str(e)}")
        st.info("💡 Try using a different browser or check internet connection")
        return False

def transcribe_audio():
    """Capture and transcribe audio"""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... speak now")
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            return text
    except:
        st.error("❌ Could not transcribe audio. Try again or type instead.")
        return None

# ==============================
# LOAD RAG PIPELINE
# ==============================
@st.cache_resource
def load_rag_pipeline(pdf_name):
    """Load and cache RAG pipeline for specific PDF"""
    pdf_path = Path("pdfs") / pdf_name
    
    with st.spinner("📖 Loading PDF..."):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.page_content = fix_pdf_text(page.page_content)
    
    with st.spinner("✂️ Splitting text..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
    
    with st.spinner("🔍 Creating embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    
    with st.spinner("🤖 Loading LLM..."):
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
# SIDEBAR
# ==============================
st.sidebar.title("⚙️ Control Panel")

# File Upload
st.sidebar.markdown("### 📤 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type="pdf")
if uploaded_file:
    pdf_folder = Path("pdfs")
    pdf_folder.mkdir(exist_ok=True)
    pdf_path = pdf_folder / uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"✅ Saved: {uploaded_file.name}")
    st.cache_resource.clear()

# PDF Selector
st.sidebar.markdown("### 📚 Select PDF")
pdf_files = get_available_pdfs()
if not pdf_files:
    st.sidebar.warning("⚠️ No PDFs uploaded yet")
    st.stop()

selected_pdf = st.sidebar.selectbox("Available PDFs:", pdf_files)

# Settings
st.sidebar.markdown("### 🎛️ Settings")
enable_voice = st.sidebar.checkbox("🎤 Enable Voice", value=False)
enable_auto_summary = st.sidebar.checkbox("📄 Auto-Summary", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Info")
st.sidebar.write(f"**Current PDF**: {selected_pdf}")
st.sidebar.write(f"**PDFs Loaded**: {len(pdf_files)}")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"**Device**: {device.upper()}")

# ==============================
# LOAD PIPELINE
# ==============================
pipeline = load_rag_pipeline(selected_pdf)

# ==============================
# MAIN DASHBOARD
# ==============================
st.title("📊 RAG Intelligence Dashboard")
st.markdown(f"**Analyzing**: `{selected_pdf}` | **Device**: {device.upper()}")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Single Query", "🔍 Cross-Query", "📄 Auto Summary", "🎤 Voice Chat", "📥 Export"])

# ==============================
# TAB 1: SINGLE QUERY WITH HISTORY
# ==============================
with tab1:
    st.markdown("### Ask Questions About Your Document")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Your question:", placeholder="Ask me anything about the document...")
    with col2:
        search_btn = st.button("🔍 Search", use_container_width=True)
    
    if search_btn and question:
        with st.spinner("⏳ Thinking..."):
            answer, context = query_rag(question, pipeline)
        
        # Save to history
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer
        }
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = load_chat_history(selected_pdf)
        st.session_state.chat_history.append(chat_entry)
        save_chat_history(selected_pdf, st.session_state.chat_history)
    
    # Load history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history(selected_pdf)
    
    # Display conversation
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        
        # Clear all button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{len(st.session_state.chat_history)} questions asked**")
        with col2:
            if st.button("🗑️ Clear All", type="secondary"):
                st.session_state.chat_history = []
                save_chat_history(selected_pdf, [])
                st.success("✅ All conversations cleared!")
                st.rerun()
        
        st.markdown("---")
        
        # Display each Q&A pair with delete option
        for i, entry in enumerate(st.session_state.chat_history):
            col1, col2, col3 = st.columns([0.05, 0.9, 0.05])
            
            with col1:
                st.markdown(f"**Q{i+1}**")
            
            with col2:
                st.markdown(f"**Your Question**: {entry['question']}")
                st.info(f"**Answer**: {entry['answer']}")
                
                # Show timestamp
                timestamp = datetime.fromisoformat(entry['timestamp'])
                st.caption(f"🕒 {timestamp.strftime('%H:%M:%S')}")
                
                if enable_voice and st.button(f"🔊 Speak Answer {i+1}", key=f"speak_{i}"):
                    speak_text(entry['answer'])
            
            with col3:
                # Delete button for this entry
                if st.button("❌", key=f"delete_{i}", help="Delete this Q&A pair"):
                    # Remove the entry
                    st.session_state.chat_history.pop(i)
                    save_chat_history(selected_pdf, st.session_state.chat_history)
                    st.success(f"✅ Deleted Q{i+1}")
                    st.rerun()
            
            st.markdown("---")
    else:
        st.info("👆 Ask a question to get started!")

# ==============================
# TAB 2: CROSS-QUERY (MULTI-PDF)
# ==============================
with tab2:
    st.markdown("### 🔍 Compare Across Documents")
    
    comparing_pdfs = st.multiselect("Select PDFs to compare:", pdf_files, default=[selected_pdf])
    compare_question = st.text_input("Ask the same question across documents:", placeholder="e.g., What are the main skills?")
    
    if st.button("📊 Compare") and compare_question and comparing_pdfs:
        results = {}
        with st.spinner("Analyzing all documents..."):
            for pdf in comparing_pdfs:
                pipeline_pdf = load_rag_pipeline(pdf)
                answer, _ = query_rag(compare_question, pipeline_pdf)
                results[pdf] = answer
        
        st.markdown("### Results Comparison")
        cols = st.columns(len(results))
        for col, (pdf, answer) in zip(cols, results.items()):
            with col:
                st.markdown(f"#### {pdf}")
                st.info(answer)
    else:
        st.info("👆 Select PDFs and ask a question to compare!")

# ==============================
# TAB 3: AUTO SUMMARY
# ==============================
with tab3:
    st.markdown("### 📄 Auto-Generate Summary")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        summary_type = st.selectbox("Summary type:", ["Key Points", "Full Summary", "Executive Brief"])
    with col2:
        summarize_btn = st.button("📝 Generate", use_container_width=True)
    
    if summarize_btn:
        summary_prompts = {
            "Key Points": "List the 5 most important key points from this document.",
            "Full Summary": "Provide a comprehensive summary of the entire document.",
            "Executive Brief": "Create a brief executive summary in 2-3 sentences."
        }
        
        with st.spinner("Generating summary..."):
            answer, context = query_rag(summary_prompts[summary_type], pipeline)
        
        st.success("✅ Summary Generated")
        st.markdown(answer)
        
        if enable_voice:
            if st.button("🔊 Listen to Summary"):
                speak_text(answer)
    else:
        st.info("👆 Click 'Generate' to create a summary!")

# ==============================
# TAB 4: VOICE CHAT
# ==============================
with tab4:
    st.markdown("### 🎤 Voice-Activated Q&A")
    
    if enable_voice:
        st.markdown("#### 🎙️ Speak Your Question")
        col1, col2 = st.columns(2)
        
        voice_question = None
        
        with col1:
            if st.button("🎙️ Start Recording", use_container_width=True):
                with st.spinner("🎤 Listening... (speak now)"):
                    try:
                        recognizer = sr.Recognizer()
                        with sr.Microphone() as source:
                            st.info("🎤 Say your question now...")
                            audio = recognizer.listen(source, timeout=5)
                            voice_question = recognizer.recognize_google(audio)
                            st.success(f"✅ Heard: '{voice_question}'")
                    except sr.WaitTimeoutError:
                        st.error("❌ No speech detected. Try again.")
                    except sr.UnknownValueError:
                        st.error("❌ Could not understand speech. Try again.")
                    except sr.RequestError as e:
                        st.error(f"❌ Speech recognition error: {e}")
                        st.info("💡 Check your internet connection")
                    except Exception as e:
                        st.error(f"❌ Microphone error: {e}")
                        st.info("💡 Make sure your browser has microphone permissions")
        
        with col2:
            st.markdown("#### 🔊 Voice Settings")
            voice_speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
            voice_lang = st.selectbox("Language", ["en", "es", "fr", "de"], index=0)
            
            if st.button("🔊 Test Voice", use_container_width=True):
                test_text = "Hello! Voice synthesis is working correctly."
                if speak_text(test_text):
                    st.success("✅ Voice test successful!")
        
        # Process voice question
        if voice_question:
            st.markdown("---")
            st.markdown("#### 📝 Processing Your Voice Question")
            with st.spinner("⏳ Analyzing..."):
                answer, _ = query_rag(voice_question, pipeline)
            
            st.markdown(f"**Your Question**: {voice_question}")
            st.info(f"**Answer**: {answer}")
            
            # Auto-play answer
            if st.button("🔊 Play Answer", use_container_width=True):
                speak_text(answer)
            
            # Save to history
            chat_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": f"[🎤 VOICE] {voice_question}",
                "answer": answer
            }
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = load_chat_history(selected_pdf)
            st.session_state.chat_history.append(chat_entry)
            save_chat_history(selected_pdf, st.session_state.chat_history)
            
    else:
        st.warning("🎤 Voice feature is disabled. Enable it in Settings!")
        st.info("💡 Voice requires internet connection for speech recognition")

# ==============================
# TAB 5: EXPORT
# ==============================
with tab5:
    st.markdown("### 📥 Export Conversation")
    
    if "chat_history" in st.session_state and st.session_state.chat_history:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export as Markdown"):
                markdown_content = f"# Conversation Export\n\n**PDF**: {selected_pdf}\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n"
                for i, entry in enumerate(st.session_state.chat_history):
                    markdown_content += f"## Q{i+1}: {entry['question']}\n\n"
                    markdown_content += f"**Answer**: {entry['answer']}\n\n---\n\n"
                
                st.download_button(
                    label="⬇️ Download Markdown",
                    data=markdown_content,
                    file_name=f"conversation_{selected_pdf.replace('.pdf', '')}.md",
                    mime="text/markdown"
                )
        
        with col2:
            if st.button("📑 Export as Text"):
                text_content = f"CONVERSATION EXPORT\n"
                text_content += f"PDF: {selected_pdf}\n"
                text_content += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                text_content += "="*50 + "\n\n"
                
                for i, entry in enumerate(st.session_state.chat_history):
                    text_content += f"Q{i+1}: {entry['question']}\n"
                    text_content += f"Answer: {entry['answer']}\n"
                    text_content += "-"*50 + "\n\n"
                
                st.download_button(
                    label="⬇️ Download Text",
                    data=text_content,
                    file_name=f"conversation_{selected_pdf.replace('.pdf', '')}.txt",
                    mime="text/plain"
                )
        
        st.markdown("---")
        st.markdown(f"### 📊 Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", len(st.session_state.chat_history))
        with col2:
            avg_answer_len = sum(len(e['answer'].split()) for e in st.session_state.chat_history) // max(1, len(st.session_state.chat_history))
            st.metric("Avg Answer Length", f"{avg_answer_len} words")
        with col3:
            st.metric("Current PDF", selected_pdf.split('.')[0])
    else:
        st.info("💭 No conversation to export yet. Ask some questions first!")
