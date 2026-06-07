#ifndef OPENCL_KMEANS_H
#define OPENCL_KMEANS_H

// ============================================================
//  K-Means Clustering — Versi OpenCL (paralelisasi GPU/CPU)
// ============================================================

// Jalankan K-Means dengan jarak Mahalanobis menggunakan OpenCL.
// Assignment step dijalankan di GPU via kernel, update di CPU.
//
// Parameter sama dengan versi sekuensial.
// kernelPath: path ke file kernel.cl
void kmeansOpenCL(const float* pixels, int N, int K,
                  float* centroids, int* labels,
                  int maxIter, const float* invCov,
                  const char* kernelPath = "kernel.cl");

#endif // OPENCL_KMEANS_H
