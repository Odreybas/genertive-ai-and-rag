# Generative AI & RAG System

A comprehensive **Retrieval-Augmented Generation (RAG) system** with a professional web dashboard for document analysis and Q&A.

## ✨ Features

### 🤖 Core RAG Functionality
- **Intelligent Q&A**: Ask questions about your PDF documents and get comprehensive answers
- **Multi-PDF Support**: Upload and analyze multiple PDFs simultaneously
- **Smart Context Retrieval**: Uses FAISS vector search for relevant document sections
- **Advanced LLM**: Powered by FLAN-T5 with beam search and optimized generation

### 🎛️ Professional Dashboard
- **5 Feature Tabs**: Single Query, Cross-Query, Auto Summary, Voice Chat, Export
- **File Management**: Drag-and-drop PDF upload, automatic storage
- **Conversation History**: Persistent chat history with smart deletion options
- **Voice Integration**: Text-to-speech with Google TTS for audio responses

### 🗑️ Conversation Management
- **Individual Delete**: Remove specific Q&A pairs with one click
- **Clear All**: Reset entire conversation history
- **Per-PDF History**: Separate conversations for each document

### 🔊 Voice Features
- **Speech Synthesis**: Convert answers to natural speech
- **Voice Controls**: Enable/disable voice in settings
- **Audio Playback**: Built-in audio player in browser

### 📊 Advanced Features
- **Cross-Document Query**: Compare answers across multiple PDFs
- **Auto Summary**: Generate key points, full summaries, or executive briefs
- **Export Options**: Download conversations as Markdown or Text files
- **Statistics**: Track query counts and response metrics

## 🚀 Quick Start

### **Option 1: One-Click Start (Easiest)**
```bash
./start.sh
```
*Automatically installs dependencies and opens dashboard*

### **Option 2: Python Launcher**
```bash
./start.py
```
*Python script that handles everything automatically*

### **Option 3: Manual Start**
```bash
# Install dependencies (if needed)
pip install -r requirements.txt

# Start dashboard
streamlit run dashboard.py --server.port=7861
```

### **Option 4: Direct Command**
```bash
streamlit run dashboard.py --server.port=7861
```

### **Access Dashboard**
Open `http://localhost:7861` in your browser

---

## 📋 **Quick Reference**

| What | Command |
|------|---------|
| **Start Dashboard** | `./start.sh` or `./start.py` |
| **Install Dependencies** | `pip install -r requirements.txt` |
| **Manual Start** | `streamlit run dashboard.py --server.port=7861` |
| **Stop Dashboard** | `Ctrl+C` in terminal |
| **Check Status** | Visit `http://localhost:7861` |

**💡 Pro Tip**: Just remember `./start.sh` or `./start.py` - they do everything automatically!

## 📁 Project Structure

```
genertive-ai-and-rag/
├── dashboard.py          # Main Streamlit dashboard
├── main.py              # CLI version (legacy)
├── web_interface.py     # Simple web interface (legacy)
├── start.sh             # 🚀 Easy bash launcher
├── start.py             # 🚀 Easy Python launcher
├── requirements.txt     # Python dependencies
├── README.md           # This documentation
├── pdfs/               # PDF storage directory
├── chat_history/       # Conversation history (auto-created)
└── venv/               # Virtual environment (optional)
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit (web dashboard)
- **Backend**: Python 3.12+
- **LLM**: Google FLAN-T5 Large (via Transformers)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Voice**: Google Text-to-Speech (gTTS)
- **PDF Processing**: PyPDF
- **UI Components**: Streamlit tabs, columns, expanders

## 🎯 Usage Examples

### Basic Q&A
1. Upload a PDF (CV, report, manual, etc.)
2. Ask: "What are the main skills mentioned?"
3. Get comprehensive answer with context

### Cross-Document Analysis
1. Upload multiple PDFs
2. Use Cross-Query tab
3. Ask: "Compare the experience levels"
4. See side-by-side answers

### Voice Interaction
1. Enable voice in sidebar settings
2. Ask questions or generate summaries
3. Click 🔊 to hear responses

### Export Conversations
1. Build up conversation history
2. Go to Export tab
3. Download as Markdown or Text

## 🔧 Configuration

### Voice Settings
- Enable/disable in sidebar
- Works best in Chrome/Firefox
- Requires internet for TTS

### Model Settings
- Automatically uses GPU if available
- Falls back to CPU otherwise
- Models cached after first load

### File Management
- PDFs stored in `pdfs/` folder
- Conversations saved per PDF
- Automatic cleanup of temp files

## 📊 Performance

- **First Load**: ~2-3 minutes (model download)
- **Subsequent Loads**: Instant (cached models)
- **Memory Usage**: ~4GB RAM for FLAN-T5 Large
- **Response Time**: 5-15 seconds per query

## 🐛 Troubleshooting

### Common Issues
- **Port busy**: Use different port (`--server.port=7862`)
- **Voice not working**: Check internet connection
- **Import errors**: Run `pip install -r requirements.txt`
- **Memory issues**: Use smaller model or CPU mode

### Reset Everything
```bash
# Clear all data
rm -rf chat_history/ pdfs/*.pdf
# Restart dashboard
streamlit run dashboard.py
```

## 📈 Future Enhancements

- [ ] REST API endpoints
- [ ] User authentication
- [ ] Batch PDF processing
- [ ] Custom model fine-tuning
- [ ] Mobile app integration
- [ ] Multi-language support

## 🤝 Contributing

This project demonstrates advanced RAG implementation with modern web interfaces. Feel free to fork and extend!

## 📄 License

Built for educational and professional document analysis purposes.

---

**🎉 Your RAG system is complete and ready for production use!**