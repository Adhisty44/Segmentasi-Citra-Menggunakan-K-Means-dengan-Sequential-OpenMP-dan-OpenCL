// ============================================================
//  main.cpp — Driver Utama: Benchmark K-Means Segmentasi Citra
//
//  Menjalankan 3 versi K-Means (Sequential, OpenMP, OpenCL)
//  dan membandingkan waktu eksekusi masing-masing.
//
//  Penggunaan:
//    ./kmeans_benchmark <path_gambar> [ukuran_resize]
//
//  Contoh:
//    ./kmeans_benchmark sawah6.png 100
//    ./kmeans_benchmark sawah6.png 500
// ============================================================

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include <omp.h>
#include "opencl_kmeans.h"
#include "openmp_kmeans.h"
#include "sequential.h"
#include "utils.h"
using namespace std;

// ============================================================
//  Konfigurasi
// ============================================================
static const int K = 5;          // Jumlah cluster
static const int MAX_ITER = 300; // Maksimum iterasi
static const int SEED = 42;      // Seed random untuk centroid

int main(int argc, char **argv) {
  // ---- Parse Argumen ----
  if (argc < 2) {
    cerr << "Penggunaan: " << argv[0] << " <path_gambar> [ukuran_resize]"
              << endl;
    cerr << "Contoh:     " << argv[0] << " sawah6.png 100" << endl;
    return 1;
  }

  string imagePath = argv[1];
  int targetSize = (argc >= 3) ? atoi(argv[2]) : 100;

   cout << "\n";
   cout << "============================================================"
            <<  endl;
   cout << "   BENCHMARK K-MEANS SEGMENTASI CITRA (Mahalanobis)"
            <<  endl;
   cout << "============================================================"
            <<  endl;
   cout << "  Gambar      : " << imagePath <<  endl;
   cout << "  Ukuran      : " << targetSize << " x " << targetSize
            <<  endl;
   cout << "  K (cluster) : " << K <<  endl;
   cout << "  Max Iterasi : " << MAX_ITER <<  endl;
   cout << "============================================================\n"
            <<  endl;

  // ---- 1. Load & Preprocess ----
   cout << "[1/5] Loading dan preprocessing gambar..." <<  endl;
  ImageData imgData = loadAndPreprocess(imagePath, targetSize);
  int N = imgData.N;
   cout << "  Total piksel: " << N << " (" << imgData.width << "x"
            << imgData.height << ")" <<  endl;

  // ---- 2. Hitung Covariance & Inverse ----
   cout << "[2/5] Menghitung covariance matrix & inverse..." <<  endl;
  float cov[9], invCov[9];
  computeCovariance(imgData.pixels.data(), N, cov);
  if (!invertMatrix3x3(cov, invCov)) {
    cerr << "[ERROR] Covariance matrix singular!" <<  endl;
    return 1;
  }
   cout << "  Covariance matrix OK.\n" <<  endl;

  // ---- 3. Inisialisasi Centroid Awal ----
  // Semua versi mulai dari centroid yang SAMA (fair comparison)
  float initCentroidsArr[K * 3];
  initCentroids(imgData.pixels.data(), N, K, initCentroidsArr, SEED);

   cout << "  Centroid awal:" <<  endl;
  for (int i = 0; i < K; i++) {
     cout << "    C" << i << " = (" << initCentroidsArr[i * 3 + 0] << ", "
              << initCentroidsArr[i * 3 + 1] << ", "
              << initCentroidsArr[i * 3 + 2] << ")" <<  endl;
  }
   cout <<  endl;

  // Array untuk menyimpan hasil dan timing
  float centroids_seq[K * 3], centroids_omp[K * 3], centroids_ocl[K * 3];
  int *labels_seq = new int[N];
  int *labels_omp = new int[N];
  int *labels_ocl = new int[N];
  double time_seq = 0, time_omp = 0, time_ocl = 0;

  // ============================================================
  //  BENCHMARK 1: Sequential
  // ============================================================
  {
    cout << "[3/5] Menjalankan K-Means SEQUENTIAL..." <<  endl;
    memcpy(centroids_seq, initCentroidsArr, K * 3 * sizeof(float));

    auto start =  chrono::high_resolution_clock::now();
    kmeansSequential(imgData.pixels.data(), N, K, centroids_seq, labels_seq,
                     MAX_ITER, invCov);
    auto end =  chrono::high_resolution_clock::now();

    time_seq =  chrono::duration<double,  milli>(end - start).count();
    cout << "  Selesai dalam " <<  fixed <<  setprecision(2)
              << time_seq << " ms" <<  endl;

    // Simpan hasil
    saveResult("result_sequential.png", centroids_seq, labels_seq,
               imgData.width, imgData.height);
  }

  // ============================================================
  //  BENCHMARK 2: OpenMP
  // ============================================================
  {
    cout << "\n[4/5] Menjalankan K-Means OPENMP..." <<  endl;
    memcpy(centroids_omp, initCentroidsArr, K * 3 * sizeof(float));

    auto start =  chrono::high_resolution_clock::now();
    kmeansOpenMP(imgData.pixels.data(), N, K, centroids_omp, labels_omp,
                 MAX_ITER, invCov);
    auto end =  chrono::high_resolution_clock::now();

    time_omp =  chrono::duration<double,  milli>(end - start).count();
    cout << "  Selesai dalam " <<  fixed <<  setprecision(2)
              << time_omp << " ms" <<  endl;

    saveResult("result_openmp.png", centroids_omp, labels_omp, imgData.width,
               imgData.height);
  }

  // ============================================================
  //  BENCHMARK 3: OpenCL
  // ============================================================
  {
    cout << "\n[5/5] Menjalankan K-Means OPENCL..." <<  endl;
    memcpy(centroids_ocl, initCentroidsArr, K * 3 * sizeof(float));

    auto start =  chrono::high_resolution_clock::now();
    kmeansOpenCL(imgData.pixels.data(), N, K, centroids_ocl, labels_ocl,
                 MAX_ITER, invCov);
    auto end =  chrono::high_resolution_clock::now();

    time_ocl =  chrono::duration<double,  milli>(end - start).count();
    cout << "  Selesai dalam " <<  fixed <<  setprecision(2)
              << time_ocl << " ms" <<  endl;

    saveResult("result_opencl.png", centroids_ocl, labels_ocl, imgData.width,
               imgData.height);
  }

  // ============================================================
  //  TABEL BENCHMARK
  // ============================================================
  double speedup_omp = time_seq / time_omp;
  double speedup_ocl = time_seq / time_ocl;

  cout << "\n";
  cout << "============================================================"
            <<  endl;
  cout << "              HASIL BENCHMARK K-MEANS" <<  endl;
  cout << "============================================================"
            <<  endl;
  cout << "  Gambar      : " << imagePath <<  endl;
  cout << "  Ukuran      : " << targetSize << "x" << targetSize << " ("
            << N << " piksel)" <<  endl;
  cout << "  K           : " << K <<  endl;
  cout << "------------------------------------------------------------"
            <<  endl;
  cout <<  left <<  setw(16) << "  Metode" <<  right
            <<  setw(14) << "Waktu (ms)" <<  setw(12) << "Speedup"
            <<  endl;
  cout << "------------------------------------------------------------"
            <<  endl;

  cout <<  left <<  setw(16) << "  Sequential" <<  right
            <<  setw(11) <<  fixed <<  setprecision(2) << time_seq
            <<  setw(10) << "1.00x" <<  endl;

  cout <<  left <<  setw(16) << "  OpenMP" <<  right
            <<  setw(11) <<  fixed <<  setprecision(2) << time_omp
            <<  setw(10) <<  fixed <<  setprecision(2)
            << speedup_omp << "x" <<  endl;

  cout <<  left <<  setw(16) << "  OpenCL" <<  right
            <<  setw(11) <<  fixed <<  setprecision(2) << time_ocl
            <<  setw(10) <<  fixed <<  setprecision(2)
            << speedup_ocl << "x" <<  endl;

  cout << "============================================================"
            <<  endl;

  // ---- Centroid Akhir (verifikasi konsistensi) ----
  cout << "\n  Centroid Akhir (Sequential):" <<  endl;
  for (int i = 0; i < K; i++) {
     cout << "    C" << i << " = (" <<  fixed <<  setprecision(1)
              << centroids_seq[i * 3 + 0] << ", " << centroids_seq[i * 3 + 1]
              << ", " << centroids_seq[i * 3 + 2] << ")" <<  endl;
  }

  cout << "\n  Gambar hasil tersimpan:" <<  endl;
  cout << "    - result_sequential.png" <<  endl;
  cout << "    - result_openmp.png" <<  endl;
  cout << "    - result_opencl.png" <<  endl;
  cout <<  endl;

  // Cleanup
  delete[] labels_seq;
  delete[] labels_omp;
  delete[] labels_ocl;

  return 0;
}
