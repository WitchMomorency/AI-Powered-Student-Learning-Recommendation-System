import pandas as pd
import pickle
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Pastikan file .pkl berada di satu folder dengan script ini
MODEL_PATH = 'recommendation_model.pkl'

# Variabel Global untuk menyimpan data model
user_item_matrix = None
user_similarity_df = None

def load_model():
    """
    Fungsi untuk memuat model dari file pickle.
    Dijalankan sekali saat server/aplikasi dimulai.
    """
    global user_item_matrix, user_similarity_df
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"CRITICAL ERROR: File '{MODEL_PATH}' tidak ditemukan. Pastikan file ada di folder yang sama.")

    print(f"Loading model from {MODEL_PATH}...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            # Pastikan key ini sesuai dengan saat export di Jupyter Notebook
            user_item_matrix = data['user_item_matrix']
            user_similarity_df = data['user_similarity_df']
        print("Model berhasil dimuat! Siap memberikan rekomendasi.")
    except Exception as e:
        print(f"Gagal memuat model: {e}")

def get_recommendations(user_id, top_n=3):
    """
    Fungsi Utama untuk dipanggil oleh API Back-End.
    
    Args:
        user_id (str/int): ID Mahasiswa yang ingin diberi rekomendasi.
        top_n (int): Jumlah materi yang ingin direkomendasikan.
        
    Returns:
        list: Daftar nama materi (Quiz/Assignment) yang direkomendasikan.
              Mengembalikan list kosong [] jika user tidak ditemukan.
    """
    
    # 1. Pastikan model sudah terload
    if user_item_matrix is None or user_similarity_df is None:
        load_model()
        
    # 2. Cek apakah User ID ada di database model
    # Konversi user_id ke format yang sesuai (misal string/int) jika perlu
    if user_id not in user_item_matrix.index:
        print(f"Warning: UserID {user_id} tidak ditemukan dalam data training.")
        return [] # Kembalikan list kosong atau default recommendation

    # ==========================================
    # LOGIKA COLLABORATIVE FILTERING (USER-BASED)
    # ==========================================
    
    # A. Cari User lain yang mirip (Neighbors)
    # Mengambil row similarity untuk user tersebut, drop dirinya sendiri, lalu sort descending
    sim_scores = user_similarity_df[user_id].drop(user_id).sort_values(ascending=False)
    
    # Ambil Top-5 tetangga terdekat
    top_neighbors = sim_scores.head(5).index
    
    recommendations = []
    
    # B. Filter Materi
    # Ambil materi yang sudah dikerjakan user target (agar tidak direkomendasikan ulang)
    target_user_items = user_item_matrix.loc[user_id]
    completed_items = target_user_items[target_user_items > 0].index.tolist()
    
    for neighbor in top_neighbors:
        # Ambil materi milik tetangga
        neighbor_items = user_item_matrix.loc[neighbor]
        
        # Cari materi yang nilainya > 0 (sudah dikerjakan neighbor)
        potential_items = neighbor_items[neighbor_items > 0].index.tolist()
        
        for item in potential_items:
            # Syarat Rekomendasi:
            # 1. Item belum dikerjakan oleh user target
            # 2. Item belum masuk ke list rekomendasi saat ini
            if item not in completed_items and item not in recommendations:
                recommendations.append(item)
                
                # Jika kuota rekomendasi sudah terpenuhi, stop loop
                if len(recommendations) >= top_n:
                    return recommendations
    
    # Jika tidak ada rekomendasi spesifik (misal karena user sudah mengerjakan semua)
    if not recommendations:
        return ["Tidak ada rekomendasi baru (Materi lengkap)."]
    
    return recommendations

# ==========================================
# CONTOH PENGGUNAAN (MAIN BLOCK)
# Bagian ini hanya berjalan jika file ini dijalankan langsung (bukan di-import)
# Gunakan ini untuk testing cepat sebelum integrasi API.
# ==========================================
if __name__ == "__main__":
    print("--- TESTING MODE ---")
    
    # 1. Load Model
    load_model()
    
    # 2. Ambil satu contoh User ID dari data (untuk tes)
    # Kita ambil user pertama yang ada di matrix
    if user_item_matrix is not None:
        sample_user = user_item_matrix.index[0]
        print(f"\nMencoba rekomendasi untuk User ID: {sample_user}")
        
        # 3. Panggil Fungsi
        hasil = get_recommendations(sample_user, top_n=3)
        
        print("Hasil Rekomendasi:")
        print(hasil)
    else:
        print("Model gagal dimuat, tidak bisa testing.")