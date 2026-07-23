========================================================================
     PROYEK UAS MINI MACHINE LEARNING - ANALISIS & PREDIKSI NILAI
========================================================================

A. DESKRIPSI PROYEK
   Proyek ini merupakan implementasi Mini Machine Learning sederhana untuk 
   memenuhi tugas Ujian Akhir Semester (UAS) Individu. Program ini melakukan 
   Analisis Eksplorasi Data (EDA) serta pemodelan Regresi Linear Sederhana 
   dari awal (from scratch) tanpa menggunakan library machine learning eksternal 
   seperti scikit-learn. Model matematika dilatih menggunakan numpy untuk menghitung 
   slope (kemiringan) dan intercept (titik potong) garis regresi.

   Topik Proyek: Pengaruh Jam Belajar dan Bimbingan Belajar (Bimbel) terhadap Nilai Ujian.

B. STRUKTUR FOLDER
   analisis_nilai_mahasiswa/
   ├─ utils.py               <- Berisi fungsi generate dataset & matematika model regresi
   ├─ main.py                <- Program utama yang menjalankan alur pemrosesan data & plot
   ├─ Dataset/
   │  └─ dataset.csv         <- Dataset publik/tiruan berformat CSV yang dihasilkan program
   ├─ output/
   │  ├─ hasil_analisis.txt  <- Laporan analisis statistik & performa model (txt)
   │  ├─ grafik_hubungan.png <- Visualisasi Scatter Plot & garis regresi
   │  └─ grafik_bimbel.png   <- Visualisasi Bar Chart rata-rata nilai bimbel
   └─ Readme.txt             <- File dokumentasi ini

C. REQUISITE & KETERGANTUNGAN LIBRARY (LIBRARIES REQUIRED)
   Program ini berjalan menggunakan Python 3.x dan memerlukan library berikut:
   1. pandas     - Untuk membaca dan memanipulasi dataset.
   2. numpy      - Untuk perhitungan matematika model regresi linear (OLS).
   3. matplotlib - Untuk plotting grafik dasar.
   4. seaborn    - Untuk visualisasi grafik agar lebih menarik & estetis.

   Instalasi library menggunakan terminal/command prompt:
   pip install pandas numpy matplotlib seaborn

D. CARA MENJALANKAN PROGRAM (HOW TO RUN)
   1. Buka Terminal atau Command Prompt (CMD) di dalam direktori folder "analisis_nilai_mahasiswa".
   2. Jalankan perintah berikut:
      python main.py
   3. Program akan secara otomatis:
      a. Membuat folder "Dataset" dan file "dataset.csv" jika belum ada.
      b. Membaca data dan menghitung statistik deskriptif.
      c. Melatih model Regresi Linear secara manual menggunakan NumPy.
      d. Membuat folder "output".
      e. Menghasilkan grafik "grafik_hubungan.png" dan "grafik_bimbel.png" di dalam folder output.
      f. Membuat laporan teks "hasil_analisis.txt" di dalam folder output.

E. LOGIKA MATEMATIKA MODEL MACHINE LEARNING (REGRESI LINEAR)
   Model regresi linear memodelkan hubungan linear antara variabel X (Jam_Belajar)
   dan Y (Nilai_Ujian) dengan rumus:
      Y_pred = m * X + c

   Dimana nilai m (slope) dan c (intercept) dihitung manual di utils.py dengan rumus:
      m = sum((X - mean_X) * (Y - mean_Y)) / sum((X - mean_X)^2)
      c = mean_Y - (m * mean_X)

   Kinerja model diukur menggunakan R-squared (R2 Coefficient of Determination)
   untuk mengukur seberapa baik variabel X menjelaskan variabel Y.

F. FORMAT PENGUMPULAN UAS
   Mahasiswa diharapkan mengunggah:
   1. Laporan Utama format PDF (berisi identitas, screenshot code, screenshot output, 
      link github/drive, dan link video youtube).
   2. File main.py
   3. File utils.py
   4. Folder output/ (berisi hasil_analisis.txt, grafik_hubungan.png, dan grafik_bimbel.png)
   5. Readme.txt

------------------------------------------------------------------------
Catatan: Mahasiswa wajib mengisi identitas (Nama, NIM, Kelas) di dalam 
file "output/hasil_analisis.txt" setelah program dijalankan.
========================================================================
