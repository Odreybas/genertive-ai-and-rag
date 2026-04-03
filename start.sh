#!/bin/bash
# RAG Dashboard Launcher
# Easy way to start the Streamlit dashboard

echo "🚀 Starting RAG Dashboard..."
echo "📊 Opening at: http://localhost:7861"
echo ""

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt -q
fi

# Start the dashboard
streamlit run dashboard.py --server.port=7861 --server.headless=true