#ifndef OPENMP_KMEANS_H
#define OPENMP_KMEANS_H

// ============================================================
//  K-Means Clustering — Versi OpenMP (paralelisasi CPU)
// ============================================================

// Jalankan K-Means dengan jarak Mahalanobis menggunakan OpenMP.
// Assignment step dan update step di-paralelkan.
//
// Parameter sama dengan versi sekuensial.
void kmeansOpenMP(const float* pixels, int N, int K,
                  float* centroids, int* labels,
                  int maxIter, const float* invCov);

#endif // OPENMP_KMEANS_H
