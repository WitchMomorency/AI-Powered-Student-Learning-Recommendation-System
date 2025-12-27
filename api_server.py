from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import recommendation_service
import os

# ==========================================
# CONFIG FLASK
# ==========================================
# template_folder='.' artinya: "Cari file HTML di folder ini aja"
app = Flask(__name__, template_folder='.')
CORS(app)

# ==========================================
# 1. LOAD DATA CLUSTER
# ==========================================
cluster_map = {}
try:
    clusters_df = pd.read_csv('Output_User_Clusters.csv')
    cluster_map = dict(zip(clusters_df['userid'].astype(str), clusters_df['cluster']))
    print(f" [INFO] Data Cluster dimuat: {len(cluster_map)} user terdaftar.")
except Exception as e:
    print(f" [WARNING] Gagal memuat CSV Cluster: {e}")

# ==========================================
# 2. LOAD MODEL REKOMENDASI
# ==========================================
try:
    recommendation_service.load_model()
    print(" [INFO] Model ML berhasil dimuat.")
except Exception as e:
    print(f" [ERROR] Gagal memuat model: {e}")

# ==========================================
# 3. ROUTES / API
# ==========================================

@app.route('/')
def home():
    """
    Route utama. Saat user buka localhost:5000, 
    langsung tampilkan UI Front-end.
    """
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def get_recommendation():
    data = request.json
    user_id = str(data.get('user_id'))
    
    if not user_id:
        return jsonify({"error": "User ID tidak boleh kosong"}), 400

    # 1. Cek Cluster
    user_cluster = cluster_map.get(user_id, -1)
    
    cluster_desc = "Unknown"
    if user_cluster == 2:
        cluster_desc = "Active Learner"
    elif user_cluster == 0:
        cluster_desc = "Passive/At-Risk"
    elif user_cluster == 1:
        cluster_desc = "Outlier"

    # 2. Ambil Rekomendasi ML
    try:
        recs = recommendation_service.get_recommendations(int(user_id) if user_id.isdigit() else user_id)
    except Exception:
        recs = ["Materi Dasar - Pengenalan Sistem", "Panduan Akademik"]

    # 3. Kirim Balikan
    return jsonify({
        "user_id": user_id,
        "cluster_id": int(user_cluster),
        "cluster_status": cluster_desc,
        "recommendations": recs
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  AI LEARNING SYSTEM SIAP DIGUNAKAN  ")
    print("  Buka browser dan akses: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)