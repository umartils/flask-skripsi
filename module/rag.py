from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pickle

import os

class RAGChatbot:
    def __init__(self, llm_model, vector_dir):
        self.llm_model = llm_model
        self.vector_dir = vector_dir
        self._initialized = False
        
        # Inisialisasi variabel untuk mencegah error
        self.vector_store = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.text_prompt_template = None
        self.img_prompt_template = None
    
    def _ensure_initialized(self):
        """Lazy initialization - hanya load saat dibutuhkan"""
        if self._initialized:
            return
        
        EMBEDDING_NAME = "LazarusNLP/all-indobert-base-v2"
        COLLECTION_NAME = "grape_vector_store"
        BM25_INDEX_PATH = "./bm25_index.pkl"
        
        if not os.path.exists(self.vector_dir):
            print(f"Vector store directory '{self.vector_dir}' not found. RAG chatbot will not be available.")
            self._initialized = True  # Set true untuk menghindari loop
            return
        
        print(f"Vector store directory '{self.vector_dir}' found.")
        
        if os.path.exists(BM25_INDEX_PATH):
            with open(BM25_INDEX_PATH, "rb") as f:
                self.bm25_retriever = pickle.load(f)
            print("BM25 index loaded from file.")
        else:
            print(f"Warning: BM25 index file not found at {BM25_INDEX_PATH}. BM25 retriever will not be available.")
            self.bm25_retriever = None
        
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_NAME,
                model_kwargs={"device": "cpu"},  # Gunakan "cuda" jika ada GPU
                encode_kwargs={'normalize_embeddings': True}
            )

            vector_store = Chroma(
                persist_directory=self.vector_dir,
                embedding_function=embedding_model,
                # collection_name=COLLECTION_NAME
            )

            self.vector_store = vector_store
            print("RAG chatbot initialized successfully.")
            print(f"Loaded existing vectorstore with {vector_store._collection.count()} documents")
            
            # Setup retrievers
            self.vector_retriever = vector_store.as_retriever(
                search_kwargs={"k": 5}
            )
            
            if self.bm25_retriever:
                self.hybrid_retriever = EnsembleRetriever(
                    retrievers=[self.vector_retriever, self.bm25_retriever],
                    weights=[0.8, 0.2]
                )
            else:
                # Fallback hanya ke vector retriever jika BM25 gagal dimuat
                self.hybrid_retriever = self.vector_retriever
                print("Warning: Using only vector retriever as fallback.")
            
            # self.hybrid_retriever = EnsembleRetriever(
            #     retrievers=[self.vector_retriever, self.bm25_retriever],
            #     weights=[0.5, 0.5]
            # )
            
            self.text_prompt_template = ChatPromptTemplate.from_template("""
Anda adalah AI asisten yang membahas mengenai tanaman anggur di Indonesia.
Gunakan data berikut untuk menjawab:

=== KONTEN DOKUMEN ===
{context}

=== PERTANYAAN PENGGUNA ===
{question}

=== RIWAYAT PERCAKAPAN ===
{chat_history}

Berikan jawaban:
- Akurat
- Bahasa Indonesia
- Mudah dipahami oleh semua kalangan
- Berbasis gejala dan hal lain yang relevan
- Jangan membuat jawaban yang tidak ada di konteks (jika tidak ada, katakan 'Maaf, saya tidak menemukan informasi yang relevan, silakan bertanya pada sumber lain.')
""")
            
            self.img_prompt_template = ChatPromptTemplate.from_template("""
Anda adalah AI asisten yang membahas mengenai tanaman anggur dan mendeteksi penyakit
daun pada tanaman berdasarkan hasil klasifikasi gambar.
Penyakit daun yang terdeteksi dari gambar adalah: **{class_disease}**.

Tugasmu:
1. Sebutkan hasil klasifikasi gambar yaitu {class_disease} dan jelaskan secara singkat apa itu {class_disease}.
2. Sebutkan penyebab umum dan gejala yang biasanya muncul.
3. Berikan saran yang jelas dan mudah dipahami tentang apa yang sebaiknya dilakukan pengguna (misalnya langkah penanganan awal, pencegahan, atau kapan perlu berkonsultasi dengan ahli).
4. Gunakan bahasa Indonesia yang ramah, empatik, dan mudah dimengerti.
5. Jika pengguna menanyakan hal yang tidak berkaitan dengan {class_disease}, jawab dengan sopan dan ingatkan bahwa fokus kamu adalah pada penyakit ini.
6. Jangan membuat jawaban yang tidak ada di konteks (jika tidak ada, katakan 'Maaf, saya tidak menemukan informasi yang relevan, silakan bertanya pada sumber lain.')
7. Pastikan jawabanmu ringkas dan fokus membantu pengguna memahami {class_disease}.

Gunakan data berikut untuk menjawab:

=== KONTEN DOKUMEN ===
{context}

=== PERTANYAAN PENGGUNA ===
{question}

=== RIWAYAT PERCAKAPAN ===
{chat_history}

Pertanyaan pengguna: Berdasarkan penyakit yang terdeteksi yaitu {class_disease}, berikan informasi dan saran yang relevan.
Asisten:
""")
            
            self._initialized = True
            
        except Exception as e:
            print(f"Error initializing RAG chatbot: {e}")
            self.vector_store = None
            self._initialized = True  # Set true untuk menghindari loop inisialisasi
    
    def generate_response_img(self, class_disease, chat_history):
        """Generate response untuk hasil deteksi gambar"""
        """Perform hybrid search dengan RAG"""
        # Ensure initialized sebelum digunakan
        self._ensure_initialized()
        
        # Cek apakah RAG berhasil diinisialisasi
        if self.hybrid_retriever is None:
            return "Maaf, sistem RAG belum tersedia. Silakan coba lagi nanti."
        
        try:
            user_message = f"Penyakit yang terdeteksi adalah {class_disease}. Berikan informasi dan saran yang relevan."
            docs = self.hybrid_retriever.invoke(user_message)
            context = "\n\n".join([d.page_content for d in docs])

            chain = self.img_prompt_template | self.llm_model
            response = chain.invoke({
                "context": context,
                "question": user_message,
                "chat_history": chat_history,
                "class_disease": class_disease
            })
            ai_response_text = response.content
            
            return ai_response_text
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi."
        

    def hybrid_search(self, user_message, chat_history,):
        """Perform hybrid search dengan RAG"""
        # Ensure initialized sebelum digunakan
        self._ensure_initialized()
        
        # Cek apakah RAG berhasil diinisialisasi
        if self.hybrid_retriever is None:
            return "Maaf, sistem RAG belum tersedia. Silakan coba lagi nanti."
        
        try:
            docs = self.hybrid_retriever.invoke(user_message)
            context = "\n\n".join([d.page_content for d in docs])

            chain = self.text_prompt_template | self.llm_model
            response = chain.invoke({
                "context": context,
                "question": user_message,
                "chat_history": chat_history
            })
            ai_response_text = response.content
            
            return ai_response_text
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi."