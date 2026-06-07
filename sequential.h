#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

// ============================================================
//  K-Means Clustering — Versi Sekuensial (tanpa paralelisasi)
// ============================================================

// Jalankan K-Means dengan jarak Mahalanobis secara sekuensial.
//
// Parameter:
//   pixels   - array float [N*3], data piksel RGB
//   N        - jumlah piksel
//   K        - jumlah cluster
//   centroids- array float [K*3], centroid awal (akan diupdate in-place)
//   labels   - array int [N], output label tiap piksel
//   maxIter  - maksimum iterasi
//   invCov   - array float [9], invers covariance matrix 3x3 (row-major)
void kmeansSequential(const float* pixels, int N, int K,
                      float* centroids, int* labels,
                      int maxIter, const float* invCov);

#endif // SEQUENTIAL_H
