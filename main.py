import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import fungsi utilitas yang telah kita buat dari utils.py
from utils import generate_student_dataset, fit_linear_regression, predict_linear_regression, evaluate_regression

# Konfigurasi Path File
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

DATASET_FILE = os.path.join(DATASET_DIR, "dataset.csv")
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "hasil_analisis.txt")
OUTPUT_IMG_HUBUNGAN = os.path.join(OUTPUT_DIR, "grafik_hubungan.png")
OUTPUT_IMG_BIMBEL = os.path.join(OUTPUT_DIR, "grafik_bimbel.png")

def main():
    print("="*60)
    print("   PROYEK UAS MINI MACHINE LEARNING - ANALISIS NILAI SISWA   ")
    print("="*60)
    
    # 1. Pengecekan dan Pembuatan Dataset
    if not os.path.exists(DATASET_FILE):
        print("[STATUS] Dataset tidak ditemukan. Membuat dataset baru...")
        df = generate_student_dataset(DATASET_FILE)
    else:
        print("[STATUS] Dataset ditemukan. Membaca dataset...")
        df = pd.read_csv(DATASET_FILE)
        
    print(f"[INFO] Dataset memuat {df.shape[0]} baris data dan {df.shape[1]} kolom.")
    print(df.head(5))
    print("-"*60)
    
    # 2. Statistik Deskriptif Dasar
    mean_jam = df["Jam_Belajar"].mean()
    mean_nilai = df["Nilai_Ujian"].mean()
    
    # Kelompokkan berdasarkan keikutsertaan bimbel
    bimbel_group = df.groupby("Ikut_Bimbel")["Nilai_Ujian"].mean()
    
    print("[ANALISIS 1] Statistik Deskriptif:")
    print(f"- Rata-rata waktu belajar mingguan: {mean_jam:.2f} jam")
    print(f"- Rata-rata nilai ujian akhir     : {mean_nilai:.2f} / 100")
    print(f"- Rata-rata nilai ikut Bimbel     : {bimbel_group.get('Ya', 0):.2f}")
    print(f"- Rata-rata nilai TIDAK Bimbel    : {bimbel_group.get('Tidak', 0):.2f}")
    print("-"*60)
    
    # 3. Pemodelan Machine Learning (Regresi Linear Sederhana)
    # Variabel Independen (X): Jam_Belajar
    # Variabel Dependen (Y): Nilai_Ujian
    X = df["Jam_Belajar"].values
    Y = df["Nilai_Ujian"].values
    
    # Melatih Model menggunakan rumus OLS Manual di utils.py
    print("[MODELING] Melatih model Regresi Linear Sederhana...")
    m, c = fit_linear_regression(X, Y)
    
    # Melakukan Prediksi
    Y_pred = predict_linear_regression(X, m, c)
    
    # Evaluasi Performa Model
    metrics = evaluate_regression(Y, Y_pred)
    
    print(f"- Persamaan Regresi: Nilai_Ujian = ({m:.4f} * Jam_Belajar) + {c:.4f}")
    print(f"- R-squared (R2)   : {metrics['R2']:.4f} ({metrics['R2']*100:.2f}%)")
    print(f"- Mean Abs. Error  : {metrics['MAE']:.4f} poin")
    print(f"- Root MSE (RMSE)  : {metrics['RMSE']:.4f} poin")
    print("-"*60)
    
    # 4. Membuat Visualisasi Data
    print("[VISUALIZATION] Membuat grafik analisis dan menyimpan...")
    
    # Mengatur style seaborn agar visualisasi terlihat premium & modern
    sns.set_theme(style="whitegrid")
    
    # Palette warna yang senada dan estetis
    colors = ["#3498db", "#e74c3c"]
    
    # Pastikan direktori output sudah ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- PLOT 1: Scatter Plot & Regression Line ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, 
        x="Jam_Belajar", 
        y="Nilai_Ujian", 
        hue="Ikut_Bimbel", 
        palette=colors,
        alpha=0.8, 
        s=80
    )
    
    # Membuat garis regresi berdasarkan nilai prediksi model
    # Urutkan X agar garis ditarik dari kiri ke kanan dengan mulus
    sort_idx = np.argsort(X)
    plt.plot(
        X[sort_idx], 
        Y_pred[sort_idx], 
        color="#2c3e50", 
        linestyle="--", 
        linewidth=2.5, 
        label=f"Garis Regresi (y = {m:.2f}x + {c:.2f})"
    )
    
    plt.title("Hubungan Jam Belajar vs Nilai Ujian", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Waktu Belajar (Jam / Minggu)", fontsize=12)
    plt.ylabel("Nilai Ujian Akhir", fontsize=12)
    plt.legend(title="Ikut Bimbel", frameon=True, facecolor="white", edgecolor="none")
    plt.ylim(0, 110)
    plt.tight_layout()
    
    # Simpan plot 1 ke file gambar
    plt.savefig(OUTPUT_IMG_HUBUNGAN, dpi=300)
    plt.close()
    print(f"[INFO] Grafik Hubungan Belajar berhasil disimpan ke: {OUTPUT_IMG_HUBUNGAN}")
    
    # --- PLOT 2: Bar Chart Rata-rata Nilai berdasarkan Bimbel ---
    plt.figure(figsize=(8, 6))
    # Buat dataframe ringkasan untuk mempermudah plotting
    bimbel_summary = df.groupby("Ikut_Bimbel")["Nilai_Ujian"].mean().reset_index()
    
    ax = sns.barplot(
        data=bimbel_summary,
        x="Ikut_Bimbel",
        y="Nilai_Ujian",
        hue="Ikut_Bimbel",
        palette=colors,
        dodge=False,
        legend=False
    )
    
    # Menambahkan anotasi nilai angka di atas masing-masing bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center", 
            va="center", 
            xytext=(0, 9), 
            textcoords="offset points", 
            fontsize=11, 
            fontweight="bold"
        )
        
    plt.title("Rata-rata Nilai Ujian berdasarkan Keikutsertaan Bimbel", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Mengikuti Bimbingan Belajar (Bimbel)", fontsize=12)
    plt.ylabel("Rata-rata Nilai Ujian", fontsize=12)
    plt.ylim(0, 110)
    plt.tight_layout()
    
    # Simpan plot 2 ke file gambar
    plt.savefig(OUTPUT_IMG_BIMBEL, dpi=300)
    plt.close()
    print(f"[INFO] Grafik Rata-rata Bimbel berhasil disimpan ke: {OUTPUT_IMG_BIMBEL}")
    print("-"*60)
    
    # 5. Ekspor Laporan Tertulis ke hasil_analisis.txt
    print("[EXPORT] Menyimpan hasil analisis tertulis...")
    
    laporan_content = f"""============================================================
           LAPORAN HASIL RUN MACHINE LEARNING & DATA EDA
============================================================
PROYEK: ANALISIS DAN PREDIKSI NILAI MAHASISWA (REGRESI LINEAR)
MATA KULIAH: UJIAN AKHIR SEMESTER (UAS) - MINI MACHINE LEARNING

------------------------------------------------------------
IDENTITAS MAHASISWA (Silakan sesuaikan data Anda):
Nama  : [TULIS NAMA ANDA DI SINI]
NIM   : [TULIS NIM ANDA DI SINI]
Kelas : [TULIS KELAS ANDA DI SINI]
------------------------------------------------------------

1. DESKRIPSI DATASET
   - Total Sampel Data : {df.shape[0]} mahasiswa
   - Kolom yang dianalisis:
     a. Nama (Identifier mahasiswa)
     b. Jam_Belajar (Waktu belajar per minggu dalam jam - Variabel X)
     c. Ikut_Bimbel (Apakah mengikuti bimbingan belajar tambahan - Ya/Tidak)
     d. Nilai_Ujian (Nilai ujian akhir siswa skala 0-100 - Variabel Y)

2. STATISTIK DESKRIPTIF DATASET
   - Rata-rata Jam Belajar Mingguan : {mean_jam:.2f} jam
   - Rata-rata Nilai Ujian Akhir     : {mean_nilai:.2f} poin
   
   - Perbandingan Nilai berdasarkan Bimbel:
     * Siswa yang mengikuti Bimbel   : {bimbel_group.get('Ya', 0):.2f} poin
     * Siswa yang TIDAK ikut Bimbel  : {bimbel_group.get('Tidak', 0):.2f} poin
     * Selisih Peningkatan Nilai     : {bimbel_group.get('Ya', 0) - bimbel_group.get('Tidak', 0):.2f} poin

3. RUMUS DAN LOGIKA MODEL MACHINE LEARNING (REGRESI LINEAR OLS)
   Persamaan Matematika Regresi:
   Y = m * X + c
   
   Di mana:
   - Y = Nilai Ujian Akhir (Prediksi)
   - X = Jam Belajar
   - m = Koefisien Kemiringan (Slope / Bobot)
   - c = Nilai Intercept (Konstanta bias)
   
   Hasil Perhitungan Koefisien Model (di-fit manual menggunakan NumPy OLS):
   - Slope (m)     : {m:.6f}
   - Intercept (c) : {c:.6f}
   
   Persamaan Model Akhir:
   Nilai_Ujian = ({m:.4f} * Jam_Belajar) + {c:.4f}
   
   Interpretasi:
   - Nilai Intercept ({c:.4f}): Jika seorang siswa belajar selama 0 jam, nilai prediksi dasarnya adalah {c:.2f}.
   - Slope ({m:.4f}): Setiap penambahan 1 jam waktu belajar per minggu, nilai ujian siswa diprediksi akan meningkat sebesar {m:.2f} poin.

4. EVALUASI KINERJA MODEL
   - Mean Absolute Error (MAE)       : {metrics['MAE']:.4f} poin
     (Rata-rata kesalahan mutlak prediksi model dari nilai aktual)
   - Root Mean Squared Error (RMSE)   : {metrics['RMSE']:.4f} poin
     (Standar deviasi dari error prediksi model)
   - R-squared (R2 Coefficient)       : {metrics['R2']:.6f} ({metrics['R2']*100:.2f}%)
     (Model ini mampu menerangkan {metrics['R2']*100:.2f}% variabilitas nilai ujian akhir berdasarkan waktu belajar. Sisa {100 - metrics['R2']*100:.2f}% lainnya dipengaruhi oleh faktor lain di luar model).

5. KESIMPULAN ANALISIS
   a. Terdapat hubungan positif yang kuat antara waktu belajar dengan nilai ujian siswa. Semakin lama waktu belajar, semakin tinggi nilai ujian yang diperoleh.
   b. Siswa yang mengikuti program Bimbingan Belajar (Bimbel) tambahan secara rata-rata memiliki nilai ujian yang lebih tinggi dibandingkan yang tidak mengikuti bimbel.
   c. Model regresi linear sederhana ini cukup akurat dalam memprediksi nilai ujian siswa berdasarkan jam belajar dengan tingkat error (MAE) sebesar {metrics['MAE']:.2f} poin.

============================================================
Laporan ini dibuat otomatis pada: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
============================================================
"""

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(laporan_content)
        
    print(f"[INFO] Laporan tertulis berhasil diekspor ke: {OUTPUT_TXT}")
    print("="*60)
    print("   EKSEKUSI PROGRAM SELESAI DENGAN SUKSES!   ")
    print("="*60)

if __name__ == "__main__":
    main()
