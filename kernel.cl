// ============================================================
//  OpenCL Kernel: K-Means Assignment Step (Mahalanobis)
//
//  Setiap work-item memproses 1 piksel:
//    - Hitung jarak Mahalanobis ke semua K centroid
//    - Assign label = centroid terdekat
// ============================================================

__kernel void assign_labels(
    __global const float* pixels,      // [N * 3] data piksel RGB
    __global const float* centroids,   // [K * 3] posisi centroid
    __global const float* invCov,      // [9]     invers covariance 3x3
    __global int*         labels,      // [N]     output label
    const int N,
    const int K)
{
    int i = get_global_id(0);
    if (i >= N) return;

    // Load piksel ke register
    float px0 = pixels[i * 3 + 0];
    float px1 = pixels[i * 3 + 1];
    float px2 = pixels[i * 3 + 2];

    // Load invCov ke register (shared across all centroids)
    float ic00 = invCov[0], ic01 = invCov[1], ic02 = invCov[2];
    float ic10 = invCov[3], ic11 = invCov[4], ic12 = invCov[5];
    float ic20 = invCov[6], ic21 = invCov[7], ic22 = invCov[8];

    float minDist  = 1e30f;
    int   bestLabel = 0;

    for (int j = 0; j < K; j++) {
        // diff = pixel - centroid
        float d0 = px0 - centroids[j * 3 + 0];
        float d1 = px1 - centroids[j * 3 + 1];
        float d2 = px2 - centroids[j * 3 + 2];

        // temp = invCov * diff
        float t0 = ic00 * d0 + ic01 * d1 + ic02 * d2;
        float t1 = ic10 * d0 + ic11 * d1 + ic12 * d2;
        float t2 = ic20 * d0 + ic21 * d1 + ic22 * d2;

        // dist^2 = diff^T * temp
        float dist2 = d0 * t0 + d1 * t1 + d2 * t2;

        // dist = sqrt(|dist2|)  — fabs untuk stabilitas numerik
        float dist = sqrt(fabs(dist2));

        if (dist < minDist) {
            minDist   = dist;
            bestLabel = j;
        }
    }

    labels[i] = bestLabel;
}
