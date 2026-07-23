import os
import numpy as np
import pandas as pd

def generate_student_dataset(file_path):
    """
    Menghasilkan dataset akademis mahasiswa tiruan secara acak namun realistis,
    lalu menyimpannya dalam format CSV.
    """

    np.random.seed(42)
    
    n_samples = 100

    nama_list = [f"Mahasiswa_{i+1:03d}" for i in range(n_samples)]

    jam_belajar = np.random.uniform(2.0, 20.0, n_samples)

    ikutan_bimbel = np.random.choice(["Ya", "Tidak"], size=n_samples, p=[0.4, 0.6])

    bimbel_bonus = np.where(ikutan_bimbel == "Ya", 8.0, 0.0)
    noise = np.random.normal(0, 4.0, n_samples)
    
    nilai_ujian = 45.0 + (jam_belajar * 2.1) + bimbel_bonus + noise

    nilai_ujian = np.clip(nilai_ujian, 0.0, 100.0)

    jam_belajar = np.round(jam_belajar, 1)
    nilai_ujian = np.round(nilai_ujian, 0).astype(int)

    df = pd.DataFrame({
        "Nama": nama_list,
        "Jam_Belajar": jam_belajar,
        "Ikut_Bimbel": ikutan_bimbel,
        "Nilai_Ujian": nilai_ujian
    })

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    df.to_csv(file_path, index=False)
    print(f"[INFO] Dataset berhasil dibuat di: {file_path}")
    return df

def fit_linear_regression(x, y):
    """
    Melatih model Regresi Linear Sederhana menggunakan metode OLS (Ordinary Least Squares) dari scratch.
    y = m * x + c
    
    m (slope) = sum((x - mean_x) * (y - mean_y)) / sum((x - mean_x)^2)
    c (intercept) = mean_y - m * mean_x
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sum((x - mean_x) ** 2)

    m = numerator / denominator
    c = mean_y - (m * mean_x)
    
    return m, c

def predict_linear_regression(x, m, c):
    """
    Membuat prediksi menggunakan model Regresi Linear Sederhana: y = m * x + c
    """
    return (m * x) + c

def evaluate_regression(y_true, y_pred):
    """
    Menghitung metrik performa regresi: MAE, MSE, dan R-squared.
    """

    mae = np.mean(np.abs(y_true - y_pred))
    
    mse = np.mean((y_true - y_pred) ** 2)

    rmse = np.sqrt(mse)

    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
