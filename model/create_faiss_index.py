import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.docstore.document import Document

with open("C:/Users/notif/Gen_AI_Assignment/data/qa_dataset.json", "r") as f:
    qa_data = json.load(f)

documents = [Document(page_content=item["context"]) for item in qa_data]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.from_documents(documents, embeddings)
faiss_index.save_local("faiss_index")

print("✅ FAISS index saved at: model/faiss_index")
