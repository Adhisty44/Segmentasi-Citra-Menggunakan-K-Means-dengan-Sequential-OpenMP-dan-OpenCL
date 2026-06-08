#include "utils.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>
using namespace std;
using namespace cv;

// ============================================================
//  Load & Preprocess Gambar
//  - Resize ke targetSize x targetSize
//  - Gaussian blur (sigma 0.5)
//  - Konversi BGR -> RGB
//  - Output: array float RGB
// ============================================================
ImageData loadAndPreprocess(const  string& path, int targetSize) {
    Mat img =  imread(path,   IMREAD_COLOR);
    if (img.empty()) {
        cerr << "[ERROR] Tidak bisa membuka gambar: " << path <<  endl;
        exit(1);
    }

    // Resize (LANCZOS4 setara dengan LANCZOS di PIL)
    resize(img, img,   Size(targetSize, targetSize), 0, 0,   INTER_LANCZOS4);

    // Gaussian blur (radius=0.5 di Python ~ sigma=0.5)
    GaussianBlur(img, img, Size(0, 0), 0.5);

    // OpenCV default BGR -> konversi ke RGB
    cvtColor(img, img,   COLOR_BGR2RGB);

    ImageData data;
    data.width  = img.cols;
    data.height = img.rows;
    data.N      = data.width * data.height;
    data.pixels.resize(data.N * 3);

    for (int i = 0; i < data.N; i++) {
        int r = i / data.width;
        int c = i % data.width;
          Vec3b px = img.at<  Vec3b>(r, c);
        data.pixels[i * 3 + 0] = static_cast<float>(px[0]); // R
        data.pixels[i * 3 + 1] = static_cast<float>(px[1]); // G
        data.pixels[i * 3 + 2] = static_cast<float>(px[2]); // B
    }

    return data;
}

// ============================================================
//  Hitung Covariance Matrix 3x3 (row-major)
//  cov[i*3+j] = (1/(N-1)) * sum((x_i - mean_i)*(x_j - mean_j))
//  Ditambah regularisasi 1e-5 di diagonal
// ============================================================
void computeCovariance(const float* pixels, int N, float cov[9]) {
    // Hitung mean tiap channel
    double mean[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < N; i++) {
        mean[0] += pixels[i * 3 + 0];
        mean[1] += pixels[i * 3 + 1];
        mean[2] += pixels[i * 3 + 2];
    }
    mean[0] /= N;
    mean[1] /= N;
    mean[2] /= N;

    // Hitung covariance
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double sum = 0.0;
            for (int i = 0; i < N; i++) {
                sum += (pixels[i * 3 + a] - mean[a]) *
                       (pixels[i * 3 + b] - mean[b]);
            }
            cov[a * 3 + b] = static_cast<float>(sum / (N - 1));
        }
    }

    // Regularisasi diagonal (mencegah singular)
    for (int i = 0; i < 3; i++) {
        cov[i * 3 + i] += 1e-5f;
    }
}

void euclideanDistance(const float* x1, const float* x2, float& dist) {
    float sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        float d = x1[i] - x2[i];
        sum += d * d;
    }
    dist =  sqrt(sum);
}

// ============================================================
//  Invers Matrix 3x3 (Cofactor / Adjugate Method)
// ============================================================
bool invertMatrix3x3(const float in[9], float out[9]) {
    float a = in[0], b = in[1], c = in[2];
    float d = in[3], e = in[4], f = in[5];
    float g = in[6], h = in[7], k = in[8];

    float det = a * (e * k - f * h)
              - b * (d * k - f * g)
              + c * (d * h - e * g);

    if ( fabs(det) < 1e-10f) return false;

    float invDet = 1.0f / det;

    out[0] = (e * k - f * h) * invDet;
    out[1] = (c * h - b * k) * invDet;
    out[2] = (b * f - c * e) * invDet;
    out[3] = (f * g - d * k) * invDet;
    out[4] = (a * k - c * g) * invDet;
    out[5] = (c * d - a * f) * invDet;
    out[6] = (d * h - e * g) * invDet;
    out[7] = (b * g - a * h) * invDet;
    out[8] = (a * e - b * d) * invDet;

    return true;
}

// ============================================================
//  Jarak Mahalanobis antara dua vektor 3D
//  d(x,y) = sqrt( (x-y)^T * invCov * (x-y) )
// ============================================================
float mahalanobisDistance(const float* x1, const float* x2, const float* invCov) {
    float diff[3];
    diff[0] = x1[0] - x2[0];
    diff[1] = x1[1] - x2[1];
    diff[2] = x1[2] - x2[2];

    // temp = invCov * diff
    float temp[3];
    for (int i = 0; i < 3; i++) {
        temp[i] = invCov[i * 3 + 0] * diff[0]
                + invCov[i * 3 + 1] * diff[1]
                + invCov[i * 3 + 2] * diff[2];
    }

    // result = diff^T * temp
    float result = diff[0] * temp[0] + diff[1] * temp[1] + diff[2] * temp[2];

    return  sqrt( fabs(result));
}

// ============================================================
//  Inisialisasi Centroid Secara Random (Deterministic)
//  Sama seperti np.random.seed(42) di Python
// ============================================================
void initCentroids(const float* pixels, int N, int K, float* centroids, int seed) {
    mt19937 rng(seed);

    // Buat array index [0..N-1], lalu shuffle
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), rng);

    // Ambil K pertama sebagai centroid awal
    for (int i = 0; i < K; i++) {
        int idx = indices[i];
        centroids[i * 3 + 0] = pixels[idx * 3 + 0];
        centroids[i * 3 + 1] = pixels[idx * 3 + 1];
        centroids[i * 3 + 2] = pixels[idx * 3 + 2];
    }
}

// ============================================================
//  Simpan Gambar Hasil Segmentasi
//  Setiap piksel diwarnai dengan warna centroid-nya
// ============================================================
void saveResult(const  string& path, const float* centroids, const int* labels,
                int width, int height) {
    Mat img(height, width, CV_8UC3);
    int N = width * height;

    for (int i = 0; i < N; i++) {
        int r = i / width;
        int c = i % width;
        int label = labels[i];
        // RGB -> BGR (format OpenCV)
        img.at<  Vec3b>(r, c) =   Vec3b(
            static_cast<uchar>(centroids[label * 3 + 2]),  // B
            static_cast<uchar>(centroids[label * 3 + 1]),  // G
            static_cast<uchar>(centroids[label * 3 + 0])   // R
        );
    }

    imwrite(path , img);
    cout << "  [OK] Gambar tersimpan: " << path << std::endl;
}
