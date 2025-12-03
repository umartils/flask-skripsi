
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever, BM25Retriever

class RAGChatbot:
    def __init__(self, llm_model, vector_store):
        self.llm_model = llm_model
        self.vector_store = vector_store
        
        self.vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        raw = self.vector_store.get(include=["documents"])
        all_docs = raw["documents"]
        
        self.bm25_retriever = BM25Retriever.from_texts(all_docs)
        self.bm25_retriever.k = 8
        
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]
        )
        
        self.prompt_template = ChatPromptTemplate.from_template("""
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
        - Berbasis gejala + solusi praktis
        - Jangan membuat jawaban yang tidak ada di konteks (jika tidak ada, katakan 'Maaf, saya tidak menemukan informasi yang relevan, silakan bertanya pada sumber lain.')
        """)
    
    def generate_response_text(self, user_message, chat_history):
        prompt = f"""
        You are a helpful AI assistant. Use the conversation history to answer the user's question.

        Conversation History:
        {chat_history}

        New Question: {user_message}
        Assistant:"""
        ai_response_obj = self.llm_model.invoke(prompt)
        ai_response_text = ai_response_obj.content
        return ai_response_text
    
    def generate_response_img(self, class_disease):
        prompt = f"""
        Kamu adalah asisten AI medis yang ahli dalam mendeteksi penyakit daun pada tanaman berdasarkan hasil klasifikasi gambar.
        Penyakit daun yang terdeteksi dari gambar adalah: **{class_disease}**.

        Tugasmu:
        1. Sebutkan hasil klasifikasi gambar yaitu {class_disease} dan jelaskan secara singkat apa itu {class_disease}.
        2. Sebutkan penyebab umum dan gejala yang biasanya muncul.
        3. Berikan saran yang jelas dan mudah dipahami tentang apa yang sebaiknya dilakukan pengguna (misalnya langkah penanganan awal, pencegahan, atau kapan perlu berkonsultasi dengan ahli).
        4. Gunakan bahasa Indonesia yang ramah, empatik, dan mudah dimengerti.
        5. Jika pengguna menanyakan hal yang tidak berkaitan dengan {class_disease}, jawab dengan sopan dan ingatkan bahwa fokus kamu adalah pada penyakit ini.
        6. Pastikan jawabanmu ringkas dan fokus membantu pengguna memahami {class_disease}.

        Pertanyaan pengguna: Berdasarkan penyakit yang terdeteksi yaitu {class_disease}, berikan informasi dan saran yang relevan.
        Asisten:
        """
        ai_response_obj = self.llm_model.invoke(prompt)
        ai_response_text = ai_response_obj.content
        return ai_response_text

    
    def hybrid_search(self, user_message, chat_history):
        
        docs = self.hybrid_retriever.invoke(user_message)
        context = "\n\n".join([d.page_content for d in docs])

        chain = self.prompt_template | self.llm_model
        response = chain.invoke({
            "context": context,
            "question": user_message,
            "chat_history": chat_history
        })
        ai_response_text = response.content
        return ai_response_text
