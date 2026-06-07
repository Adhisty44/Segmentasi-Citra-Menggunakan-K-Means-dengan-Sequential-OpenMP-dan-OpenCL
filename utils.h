#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// ============================================================
//  Struktur data untuk menyimpan pixel gambar (RGB, float)
// ============================================================
struct ImageData {
    std::vector<float> pixels;  // N * 3 (R,G,B flattened)
    int width;
    int height;
    int N;                      // total piksel = width * height
};

// ============================================================
//  Fungsi utilitas bersama (shared)
// ============================================================

// Load gambar, resize, Gaussian blur, konversi ke array float RGB
ImageData loadAndPreprocess(const std::string& path, int targetSize = 100);

// Hitung covariance matrix 3x3 dari data piksel (row-major)
void computeCovariance(const float* pixels, int N, float cov[9]);

// Invers matrix 3x3 (cofactor method). Return false jika singular.
bool invertMatrix3x3(const float in[9], float out[9]);

// Hitung jarak Mahalanobis antara dua vektor 3D
float mahalanobisDistance(const float* x1, const float* x2, const float* invCov);

// Inisialisasi centroid secara random (deterministic dengan seed)
void initCentroids(const float* pixels, int N, int K, float* centroids, int seed = 42);

// Simpan gambar hasil segmentasi
void saveResult(const std::string& path, const float* centroids, const int* labels,
                int width, int height);

#endif // UTILS_H
