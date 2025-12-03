from flask import Flask, request, jsonify, g, send_from_directory
import uuid
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from werkzeug.utils import secure_filename
from module.db import Database
from module.yolo import YOLODetector
from module.cnn import CNNDetector
from module.rag import RAGChatbot

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

CNN_MODEL_PATH = "model/best_fusion_model(light)_(2025-11-06_18-17).h5"
YOLO_MODEL_PATH = "model/yolo.pt"

DATABASE_FILE = 'database.db'
LLM_NAME = "gemini-2.5-flash"
VECTOR_STORE_DIR = "vectorstore_chroma_db1"

# ===== INISIASI MODEL DETEKSI =========
cnn_detector = CNNDetector(CNN_MODEL_PATH)
yolo_detector = YOLODetector(model_path="model/yolo.pt", output_dir=PROCESSED_FOLDER)

# ===== INISIASI CHATBOT =========
chatbot_model = ChatGoogleGenerativeAI(
    model=LLM_NAME,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
rag_chatbot = RAGChatbot(chatbot_model, VECTOR_STORE_DIR)

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

def allowed_file(filename):
    """Cek apakah ekstensi file diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return "Selamat datang di Chatbot API!"

@app.route("/upload_img", methods=['POST'])
def upload_image():
    db_conn = get_db()
    
    # Periksa apakah ada file di request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    session_id = request.form.get('session_id') # Ambil session_id dari form data
    room_id = request.form.get('room_id') # Ambil room_id dari form data
    
    if image_file.filename == '':
       return jsonify({'error': 'No filename provided'}), 400

    if not session_id or not room_id or image_file.filename == '':
        return jsonify({'error': 'Session ID or filename is missing'}), 400
    
    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: jpg, jpeg, png, webp'}), 400

    try:
         # Bersihkan nama file (hapus spasi dan karakter aneh)
        clean_name = secure_filename(image_file.filename)
        clean_name = clean_name.replace(" ", "_")  # Ganti spasi dengan underscore
        
        filename = f"{uuid.uuid4().hex}_{clean_name}"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(upload_path)

        class_name, confidence = cnn_detector.detect_objects(upload_path)
        if confidence > 0.45:
        # 🔥 Jalankan deteksi YOLO
            user_message = f"Gambar telah diproses menggunakan CNN dengan hasil prediksi kelas: {class_name} dengan confidence: {confidence:.2f}."
            db_conn.save_message(session_id, room_id, 'user', user_message)
            processed_path = yolo_detector.detect_objects(upload_path, filename_prefix="detected")
            
            chat_history_from_db = db_conn.get_history(session_id, room_id)

            # 3. Format history dan prompt (tidak ada perubahan di sini)
            formatted_history = []
            for role, msg in chat_history_from_db:
                if role == 'user':
                    formatted_history.append(f"Human: {msg}")
                elif role == 'ai':
                    formatted_history.append(f"Assistant: {msg}")

            rag_context = "\n".join(formatted_history)
            # Buat URL hasil
            processed_url = f"http://{request.host}/processed/{os.path.basename(processed_path)}"
            ai_response_text = rag_chatbot.generate_response_img(class_name, rag_context)
            db_conn.save_message(session_id, room_id, 'ai', ai_response_text)
            
            return jsonify({
                "response": f"Gambar telah diproses menggunakan YOLO dan CNN dengan kelas: {class_name} dengan confidence: {confidence:.2f}.",
                "image_url": processed_url,
                "ai_response": ai_response_text
            }), 200
        else:
            return jsonify({
                "response": f"Gambar tidak dikenali sebagai penyakit tanaman.",
                "image_url": None,
                "ai_response": "Mohon unggah gambar yang jelas dari daun tanaman yang menunjukkan gejala penyakit untuk analisis lebih lanjut."
            }), 200
        
        # Simpan balasan AI ke database

    except Exception as e:
        print(f"An error occurred during image upload: {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500

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

@app.route('/processed/<filename>')
def serve_uploaded_image(filename):
    return send_from_directory('processed', filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)