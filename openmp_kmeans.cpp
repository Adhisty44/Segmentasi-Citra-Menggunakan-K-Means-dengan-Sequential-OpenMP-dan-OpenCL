#include "openmp_kmeans.h"
#include "utils.h"
#include <omp.h>
#include <cstring>
#include <cmath>
#include <iostream>
using namespace std;

// ============================================================
//  K-Means OpenMP — Implementasi
//
//  Paralelisasi:
//    - Assignment step: #pragma omp parallel for
//      Setiap thread memproses subset piksel secara independen
//    - Update step: setiap thread punya accumulator lokal,
//      digabung lewat critical section
// ============================================================
void kmeansOpenMP(const float* pixels, int N, int K,
                  float* centroids, int* labels,
                  int maxIter, const float* invCov) {

    cout << "  [OpenMP] Jumlah thread yang digunakan : " << omp_get_max_threads() << " thread" << endl;

    float* prevCentroids = new float[K * 3];

    for (int iter = 0; iter < maxIter; iter++) {
        // Simpan centroid sebelumnya
        memcpy(prevCentroids, centroids, K * 3 * sizeof(float));

        // === STEP 1: Assignment (PARALEL) ===
        #pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < N; i++) {
            float minDist = 1e30f;
            int   bestLabel = 0;

            for (int j = 0; j < K; j++) {
                float dist = mahalanobisDistance(
                    &pixels[i * 3], &centroids[j * 3], invCov);
                if (dist < minDist) {
                    minDist   = dist;
                    bestLabel = j;
                }
            }
            labels[i] = bestLabel;
        }

        // === STEP 2: Update Centroids (PARALEL) ===
        // Setiap thread punya sums & counts lokal → gabung di akhir
        float* globalSums  = new float[K * 3]();
        int*   globalCounts = new int[K]();

        #pragma omp parallel
        {
            // Accumulator lokal per-thread
            float* localSums  = new float[K * 3]();
            int*   localCounts = new int[K]();

            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                int lbl = labels[i];
                localSums[lbl * 3 + 0] += pixels[i * 3 + 0];
                localSums[lbl * 3 + 1] += pixels[i * 3 + 1];
                localSums[lbl * 3 + 2] += pixels[i * 3 + 2];
                localCounts[lbl]++;
            }

            // Gabungkan accumulator lokal ke global
            #pragma omp critical
            {
                for (int j = 0; j < K * 3; j++)
                    globalSums[j] += localSums[j];
                for (int j = 0; j < K; j++)
                    globalCounts[j] += localCounts[j];
            }

            delete[] localSums;
            delete[] localCounts;
        }

        // Hitung centroid baru
        for (int j = 0; j < K; j++) {
            if (globalCounts[j] > 0) {
                centroids[j * 3 + 0] = globalSums[j * 3 + 0] / globalCounts[j];
                centroids[j * 3 + 1] = globalSums[j * 3 + 1] / globalCounts[j];
                centroids[j * 3 + 2] = globalSums[j * 3 + 2] / globalCounts[j];
            }
        }

        delete[] globalSums;
        delete[] globalCounts;

        // === STEP 3: Cek Konvergensi ===
        bool converged = true;
        for (int j = 0; j < K * 3; j++) {
            if ( fabs(centroids[j] - prevCentroids[j]) > 1e-6f) {
                converged = false;
                break;
            }
        }

        if (converged) break;
    }

    delete[] prevCentroids;
}
