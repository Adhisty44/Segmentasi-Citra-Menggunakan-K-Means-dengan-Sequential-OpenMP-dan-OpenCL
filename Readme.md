# Segmentasi Citra Menggunakan K-Means dengan Sequential, OpenMP, dan OpenCL

## Anggota Kelompok

* Adhistya Wildan Maulana Akbar ; 25032014015
* Michael Frearjuna             ; 25032014001
* Habib Farah Adi Maskur        ; 25032014034

---

## Deskripsi Proyek

Proyek ini mengimplementasikan algoritma K-Means untuk segmentasi citra dengan menggunakan tiga pendekatan komputasi yang berbeda, yaitu:

1. Sequential (Sekuensial)
2. OpenMP (Paralel CPU)
3. OpenCL (Paralel GPU)

Setiap piksel pada citra akan dikelompokkan ke dalam beberapa cluster berdasarkan kemiripan warna menggunakan Mahalanobis Distance. Hasil segmentasi kemudian dibandingkan dari sisi performa dan waktu eksekusi untuk melihat pengaruh penggunaan komputasi paralel terhadap proses clustering.

Tujuan utama proyek ini adalah menganalisis perbedaan performa antara komputasi sekuensial, paralel CPU, dan paralel GPU pada proses segmentasi citra menggunakan algoritma K-Means.

---

## Fitur Utama

* Segmentasi citra menggunakan algoritma K-Means
* Penggunaan Mahalanobis Distance sebagai metrik jarak
* Implementasi Sequential (Single Thread)
* Implementasi OpenMP (Multi-thread CPU)
* Implementasi OpenCL (GPU Computing)
* Benchmark waktu eksekusi
* Perbandingan performa antar metode
* Penyimpanan hasil segmentasi citra

---

## Struktur Proyek

```
projek_akhir/
│
├── main.cpp
├── sequential.cpp
├── openmp_kmeans.cpp
├── opencl_kmeans.cpp
├── utils.cpp
├── kernel.cl
├── Makefile
│
├── sawah5.jpg
├── sawah6.jpg
├── Tuberculosis-224.jpg
│
└── README.md
```

---

## System Requirements

* Compiler C++ (GCC/G++)
* OpenCV
* OpenMP
* OpenCL SDK
* Sistem Operasi Windows atau Linux

---

## Cara Kompilasi

Jika menggunakan Makefile:

```
make
```

Perintah tersebut akan menghasilkan file executable:

```
kmeans_benchmark
```

atau

```
kmeans_benchmark.exe
```

pada sistem Windows.

---

## Cara Menjalankan Program

Format umum:

```bash
./kmeans_benchmark <nama_gambar> <jumlah_iterasi>
```

Contoh:

```bash
./kmeans_benchmark sawah6.jpg 100
```

atau pada Windows:

```bash
kmeans_benchmark.exe sawah6.jpg 100
```

Keterangan parameter:

* `<nama_gambar>` : gambar yang akan diproses
* `<jumlah_iterasi>` : jumlah maksimum iterasi K-Means

---

## Hasil Program

Program akan menghasilkan:

* Hasil segmentasi citra menggunakan K-Means
* Waktu eksekusi Sequential
* Waktu eksekusi OpenMP
* Waktu eksekusi OpenCL
* Perbandingan performa antar metode

---

## Analisis yang Dilakukan

Proyek ini melakukan analisis terhadap:

* Perbandingan waktu eksekusi Sequential, OpenMP, dan OpenCL
* Pengaruh paralelisme terhadap performa algoritma K-Means
* Speedup yang diperoleh dari OpenMP dan OpenCL
* Overhead komputasi dan transfer data CPU-GPU
* Kelebihan dan keterbatasan masing-masing pendekatan

---

## Link Video Presentasi

https://youtube.com/xxxxxxxx

---

## Link Repository GitHub

https://github.com/xxxxxxxx
