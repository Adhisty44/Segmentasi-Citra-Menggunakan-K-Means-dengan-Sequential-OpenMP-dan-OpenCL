#include "sequential.h"
#include "utils.h"
#include <cstring>
#include <cmath>
using namespace std;

// ============================================================
//  K-Means Sekuensial — Implementasi
//
//  Algoritma:
//    1. Assignment: untuk setiap piksel, hitung jarak Mahalanobis
//       ke semua centroid, pilih yang terdekat.
//    2. Update: hitung rata-rata piksel per cluster → centroid baru.
//    3. Cek konvergensi: jika centroid tidak berubah, berhenti.
// ============================================================
void kmeansSequential(const float* pixels, int N, int K,
                      float* centroids, int* labels,
                      int maxIter, const float* invCov) {

    float* prevCentroids = new float[K * 3];

    for (int iter = 0; iter < maxIter; iter++) {
        // Simpan centroid sebelumnya untuk cek konvergensi
        memcpy(prevCentroids, centroids, K * 3 * sizeof(float));

        // === STEP 1: Assignment ===
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

        // === STEP 2: Update Centroids ===
        float* sums  = new float[K * 3]();  // zero-initialized
        int*   counts = new int[K]();

        for (int i = 0; i < N; i++) {
            int lbl = labels[i];
            sums[lbl * 3 + 0] += pixels[i * 3 + 0];
            sums[lbl * 3 + 1] += pixels[i * 3 + 1];
            sums[lbl * 3 + 2] += pixels[i * 3 + 2];
            counts[lbl]++;
        }

        for (int j = 0; j < K; j++) {
            if (counts[j] > 0) {
                centroids[j * 3 + 0] = sums[j * 3 + 0] / counts[j];
                centroids[j * 3 + 1] = sums[j * 3 + 1] / counts[j];
                centroids[j * 3 + 2] = sums[j * 3 + 2] / counts[j];
            }
            // Jika cluster kosong, centroid tetap (tidak berubah)
        }

        delete[] sums;
        delete[] counts;

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
