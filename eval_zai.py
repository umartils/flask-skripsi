"""
Contoh Lengkap: Evaluasi RAG dengan RAGAS menggunakan Google Gemini
File ini siap dijalankan untuk project Anda
"""

import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings

# ==================== KONFIGURASI ====================

# GANTI DENGAN API KEY ANDA
GOOGLE_API_KEY = "AIzaSyAb1SiVyt7HGdNiByqMHDpMWX_Weq5hGA4" # <-- PASTIKAN ANDA MENGGANTI INI

# Path ke vector store
VECTOR_STORE_PATH = "./chroma_db"

# Model Gemini yang digunakan
GEMINI_MODEL = "gemini-2.5-flash"  # atau "gemini-2.5-pro"

# ==================== IMPORT RAG CHATBOT ====================

from module.rag import RAGChatbot  # Sesuaikan dengan nama file Anda
# Dummy class untuk contoh jika module.rag tidak ada
# class RAGChatbot:
#     def __init__(self, llm, vector_dir):
#         self.llm = llm
#         self.vector_dir = vector_dir
#         self.hybrid_retriever = None
#         print(f"RAGChatbot initialized with vector store at {self.vector_dir}")

#     def _ensure_initialized(self):
#         if self.hybrid_retriever is None:
#             print("Initializing dummy retriever...")
#             class DummyRetriever:
#                 def invoke(self, query):
#                     return [type('Document', (), {'page_content': f'Dummy context for {query}'})()]
#             self.hybrid_retriever = DummyRetriever()
#         print("RAG system is ready.")

#     def hybrid_search(self, question, chat_history):
#         return f"This is a dummy answer to the question: '{question}'"


# ==================== FUNGSI EVALUASI ====================

def evaluate_rag_with_gemini(
    rag_chatbot,
    questions,
    ground_truths,
    api_key=None,
    model="gemini-2.5-flash"
):
    """
    Evaluasi RAG menggunakan Gemini
    
    Args:
        rag_chatbot: Instance RAGChatbot
        questions: List pertanyaan
        ground_truths: List jawaban benar
        api_key: Google API key
        model: Model Gemini ("gemini-2.5-flash" atau "gemini-2.5-pro")
    
    Returns:
        results: Objek hasil evaluasi dari RAGAS
        dataset: Dataset yang dievaluasi
    """
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    print("="*70)
    print("RAGAS EVALUATION WITH GOOGLE GEMINI")
    print("="*70)
    print(f"Model: {model}")
    print(f"Test cases: {len(questions)}")
    print("="*70 + "\n")
    
    print("[1/5] Setting up Gemini LLM and Embeddings...")
    gemini_llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=api_key
    )
    
    EMBEDDING_NAME = "LazarusNLP/all-indobert-base-v2"
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    ragas_llm = LangchainLLMWrapper(gemini_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)
    print("✓ Gemini setup complete\n")
    
    print("[2/5] Initializing RAG system...")
    rag_chatbot._ensure_initialized()
    
    if rag_chatbot.hybrid_retriever is None:
        print("✗ Error: RAG system tidak tersedia")
        return None, None
    print("✓ RAG system ready\n")
    
    print("[3/5] Generating answers and retrieving contexts...")
    data = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': []
    }
    
    for idx, (q, gt) in enumerate(zip(questions, ground_truths), 1):
        print(f"  Processing {idx}/{len(questions)}: {q[:60]}...")
        
        try:
            docs = rag_chatbot.hybrid_retriever.invoke(q)
            context = [d.page_content for d in docs]
            answer = rag_chatbot.hybrid_search(q, chat_history="")
            
            data['question'].append(q)
            data['answer'].append(answer)
            data['contexts'].append(context)
            data['ground_truth'].append(gt)
            
        except Exception as e:
            print(f"  ✗ Error processing question {idx}: {e}")
            continue
    
    print(f"✓ Successfully processed {len(data['question'])} questions\n")
    
    print("[4/5] Creating evaluation dataset...")
    dataset = Dataset.from_dict(data)
    print("✓ Dataset created\n")
    
    print("[5/5] Running RAGAS evaluation...")
    metrics_to_use = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision
    ]
    
    for metric in metrics_to_use:
        metric.llm = ragas_llm
        metric.embeddings = ragas_embeddings
    
    try:
        results = evaluate(
            dataset,
            metrics=metrics_to_use,
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        
        print("✓ Evaluation complete!\n")
        
        # Konversi hasil ke DataFrame, lalu ambil baris pertama sebagai dictionary
        results_df = results.to_pandas()
        results_dict = results_df.to_dict('records')[0]
        
        # --- PERUBAHAN: Menampilkan hasil tanpa pengkondisian ---
        print("="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        for metric, score in results_dict.items():
            # Coba konversi ke float untuk memastikan tampilan konsisten, jika gagal tetap tampilkan aslinya
            try:
                score_val = float(score)
                print(f"{metric:.<40} {score_val:.4f}")
            except (ValueError, TypeError):
                print(f"{metric:.<40} {score}")
        print("="*70 + "\n")
        
        return results, dataset
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}\n")
        print("Troubleshooting tips:")
        print("1. Check your GOOGLE_API_KEY")
        print("2. Verify Gemini API quota")
        print("3. Try reducing the number of test cases")
        return None, None


def save_results(results, dataset, output_dir="./evaluation_results"):
    """
    Save hasil evaluasi ke file
    
    Args:
        results: Objek hasil evaluasi dari RAGAS
        dataset: Dataset yang dievaluasi
        output_dir: Directory output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Konversi hasil ke DataFrame, lalu ambil baris pertama sebagai dictionary
    results_df = results.to_pandas()
    results_dict = results_df.to_dict('records')[0]
    
    # Save summary
    df_summary = pd.DataFrame([results_dict])
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}")
    
    # Save detailed results
    df_detailed = dataset.to_pandas()
    detailed_path = os.path.join(output_dir, "evaluation_detailed.csv")
    df_detailed.to_csv(detailed_path, index=False)
    print(f"✓ Detailed results saved to: {detailed_path}")
    
    # --- PERUBAHAN: Menghilangkan penyimpanan interpretasi yang berisi pengkondisian ---
    print(f"\n✓ All results saved to: {output_dir}/\n")


# ==================== TEST CASES ====================

TEST_QUESTIONS = [
    "Apa gejala penyakit black rot pada anggur?",
    "Bagaimana cara mencegah penyakit downy mildew pada tanaman anggur?",
    # "Kapan waktu terbaik untuk pemupukan tanaman anggur?",
    # "Apa penyebab daun anggur menguning?",
    # "Bagaimana cara mengidentifikasi penyakit powdery mildew pada anggur?",
    # "Apa yang harus dilakukan jika tanaman anggur terkena penyakit leaf blight?",
    # "Bagaimana cara merawat tanaman anggur yang sehat?",
    # "Apa saja hama yang sering menyerang tanaman anggur?"
]

GROUND_TRUTHS = [
    "Gejala black rot pada anggur meliputi bercak coklat pada daun, buah mengering menjadi mumi hitam, dan lesi pada batang muda.",
    "Pencegahan downy mildew dapat dilakukan dengan menjaga sirkulasi udara yang baik, menghindari kelembaban berlebih, aplikasi fungisida preventif, dan pemangkasan daun yang terinfeksi.",
    # "Waktu terbaik pemupukan anggur adalah awal musim pertumbuhan dengan pemupukan lanjutan saat pembentukan bunga dan buah.",
    # "Daun anggur menguning dapat disebabkan oleh defisiensi nutrisi terutama nitrogen, kelebihan air, penyakit akar, atau serangan hama.",
    # "Powdery mildew dapat diidentifikasi dari lapisan putih seperti tepung pada permukaan daun, batang, dan buah anggur yang dapat menyebabkan daun mengkerut dan buah pecah.",
    # "Jika terkena leaf blight, segera buang daun yang terinfeksi, aplikasikan fungisida, dan tingkatkan sirkulasi udara di sekitar tanaman.",
    # "Perawatan tanaman anggur sehat meliputi penyiraman teratur, pemupukan berkala, pemangkasan rutin, dan monitoring hama dan penyakit.",
    # "Hama yang sering menyerang anggur antara lain tungau, thrips, kutu daun, dan ulat yang dapat merusak daun dan buah."
]


# ==================== MAIN PROGRAM ====================

def main():
    """Main function untuk menjalankan evaluasi"""
    
    print("\n" + "="*70)
    print("RAG EVALUATION SYSTEM")
    print("Project: Grape Disease Detection Chatbot")
    print("="*70 + "\n")
    
    if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        print("⚠️  WARNING: Please set your GOOGLE_API_KEY in the script!")
        print("Get your API key from: https://makersuite.google.com/app/apikey\n")
        return
    
    print("Initializing RAG Chatbot...")
    llm_model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    
    rag_chatbot = RAGChatbot(llm_model, vector_dir=VECTOR_STORE_PATH)
    print("✓ RAG Chatbot initialized\n")
    
    results, dataset = evaluate_rag_with_gemini(
        rag_chatbot=rag_chatbot,
        questions=TEST_QUESTIONS,
        ground_truths=GROUND_TRUTHS,
        api_key=GOOGLE_API_KEY,
        model=GEMINI_MODEL
    )
    
    if results and dataset:
        save_results(results, dataset)
        
        # --- PERUBAHAN: Menyederhanakan bagian rekomendasi ---
        print("="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        results_df = results.to_pandas()
        results_dict = results_df.to_dict('records')[0]
        
        # Coba hitung rata-rata, dengan penanganan jika nilai bukan angka
        numeric_scores = []
        for score in results_dict.values():
            try:
                numeric_scores.append(float(score))
            except (ValueError, TypeError):
                pass # Abaikan nilai non-numeric

        if numeric_scores:
            avg_score = sum(numeric_scores) / len(numeric_scores)
            print(f"\nAverage Score (for numeric metrics): {avg_score:.4f}\n")
        else:
            print("\nCould not calculate average score (no numeric metrics found).\n")

        print("✓ Evaluation finished. Check the './evaluation_results' directory for detailed CSV files.")
        
        print("\n" + "="*70 + "\n")
    else:
        print("✗ Evaluation failed. Please check the error messages above.")


# ==================== RUN ====================

if __name__ == "__main__":
    main()