# GenAI Q&A System

This project builds a transformer-based Q&A chatbot using Hugging Face, LangChain, and FastAPI.

## 🚀 Features
- Fine-tunes BERT on custom Q&A data
- Uses LangChain + FAISS for retrieval (RAG)
- Exposes an API with FastAPI
- Optional UI with Streamlit
- Optional: Quantization and F1 evaluation

## 📁 Structure
- `data/qa_dataset.json` – Q&A data
- `model/` – Training, inference, and optimization scripts
- `api/` – FastAPI backend
- `ui/` – Streamlit frontend

## 🧑‍💻 Setup
```bash
pip install -r requirements.txt
