import os
import json
import uuid
from glob import glob

from langchain.schema import Document
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="LazarusNLP/all-indobert-base-v2",
    model_kwargs={"device": "cuda"}   # ubah ke "cuda" jika pakai GPU
)

def save_to_chroma(documents, persist_dir="./chroma_db"):

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )

    vectordb.persist()
    print(f"[✔] Berhasil menyimpan {len(documents)} dokumen ke ChromaDB.")
    return vectordb

def clean_metadata(meta: dict) -> dict:
    cleaned = {}
    for key, value in meta.items():
        # Jika LIST → ubah ke string koma
        if isinstance(value, list):
            cleaned[key] = ", ".join([str(v) for v in value])
        # Jika dict di dalam metadata → ubah ke string JSON
        elif isinstance(value, dict):
            cleaned[key] = json.dumps(value)
        # Jika tipe data primitif → simpan apa adanya
        elif isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            # Jika tipe lain (tuple, set, object) → jadikan string
            cleaned[key] = str(value)
    return cleaned

def load_kb_files(kb_folder="./kb"):
    kb_files = glob(os.path.join(kb_folder, "*.json"))
    documents = []

    for file in kb_files:
        with open(file, "r", encoding="utf-8") as f:
            kb_data = json.load(f)

        # Pastikan file berisi list item KB
        if isinstance(kb_data, dict):
            kb_data = [kb_data]

        for item in kb_data:
            content = item.get("content", "")
            metadata = {
                "id": item.get("id", "unknown"),
                "category": item.get("category", "general"),
                "title": item.get("title", ""),
            }

            documents.append(Document(page_content=content, metadata=metadata))

    return documents

print("Memuat file KB...")
docs = load_kb_files("./data/json")

print("Menyimpan ke ChromaDB…")
save_to_chroma(docs, persist_dir="../chroma_db")