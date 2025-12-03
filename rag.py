from flask import Flask, request, jsonify, g, send_from_directory
import uuid
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

from werkzeug.utils import secure_filename
from module.db import Database
# from module.yolo import YOLODetector
# from module.cnn import CNNDetector
from module.rag import RAGChatbot

load_dotenv()

app = Flask(__name__)

DATABASE_FILE = 'database.db'
EMBEDDING_NAME = "LazarusNLP/all-indobert-base-v2"
LLM_NAME = "gemini-2.5-flash"
VECTOR_STORE_DIR = "vectorstore_chroma_db1"
COLLECTION_NAME = "grape_vector_store"

chatbot_model = ChatGoogleGenerativeAI(
    model=LLM_NAME,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

if os.path.exists(VECTOR_STORE_DIR):
    print(f"Vector store directory '{VECTOR_STORE_DIR}' found.")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_NAME,
            model_kwargs={"device": "cuda"},   # ubah ke "cuda" jika pakai GPU
            encode_kwargs={'normalize_embeddings': True}
        )

        vector_store = Chroma(
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME
        )

        rag_chatbot = RAGChatbot(chatbot_model, vector_store)
        print("RAG chatbot initialized successfully.")
        print(f"Loaded existing vectorstore with {vector_store._collection.count()} documents")
    except Exception as e:
        print(f"Error initializing RAG chatbot: {e}")
        rag_chatbot = None
else:
    print(f"Vector store directory '{VECTOR_STORE_DIR}' not found. RAG chatbot will not be available.")

def get_db():
    """
    Membuka koneksi database baru jika belum ada untuk request saat ini.
    """
    if 'db' not in g:
        g.db = Database(DATABASE_FILE)
        g.db.init_db() # Pastikan tabel ada
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    """
    Menutup koneksi database secara otomatis setelah request selesai.
    """
    db = g.pop('db', None)
    if db is not None:
        db.close()

@app.route("/chat", methods=['POST'])
def chat():
    # DIUBAH: Dapatkan koneksi database untuk request ini
    db_conn = get_db()
    
    data = request.json
    session_id = data.get('session_id')
    room_id = data.get('room_id')
    user_message = data.get('message')

    if not session_id or not user_message or not room_id:
        return jsonify({'error': 'session_id and message are required'}), 400

    try:
        # 1. Simpan pesan pengguna menggunakan koneksi saat ini
        db_conn.save_message(session_id, room_id, 'user', user_message)

        # 2. Ambil riwayat percakapan
        chat_history_from_db = db_conn.get_history(session_id, room_id)

        # 3. Format history dan prompt (tidak ada perubahan di sini)
        formatted_history = []
        for role, msg in chat_history_from_db:
            if role == 'user':
                formatted_history.append(f"Human: {msg}")
            elif role == 'ai':
                formatted_history.append(f"Assistant: {msg}")

        rag_context = "\n".join(formatted_history)
        
        ai_response_text = rag_chatbot.hybrid_search(user_message,rag_context)

        # 5. Simpan balasan AI
        db_conn.save_message(session_id, room_id, 'ai', ai_response_text)

        # 6. Kirim balasan
        return jsonify({'response': ai_response_text})

    except Exception as e:
        print(f"An error occurred during chat processing: {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)