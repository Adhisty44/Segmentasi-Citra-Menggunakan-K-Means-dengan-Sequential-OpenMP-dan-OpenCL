#include "opencl_kmeans.h"
#include "utils.h"
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <vector>
using namespace std;

// ============================================================
//  Helper: Baca file kernel.cl sebagai string
// ============================================================
static  string readKernelSource(const char* path) {
     ifstream file(path);
    if (!file.is_open()) {
        cerr << "[ERROR] Tidak bisa membuka kernel: " << path <<  endl;
        exit(1);
    }
     stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// ============================================================
//  Helper: Cek error OpenCL
// ============================================================
static void checkCL(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        cerr << "[OpenCL ERROR] " << msg << " (code: " << err << ")" <<  endl;
        exit(1);
    }
}

// ============================================================
//  K-Means OpenCL — Implementasi
//
//  Alur:
//    1. Setup OpenCL (platform, device, context, queue)
//    2. Kompilasi kernel dari file .cl
//    3. Loop iterasi:
//       a. Upload centroids ke GPU buffer
//       b. Jalankan kernel assign_labels
//       c. Download labels dari GPU
//       d. Update centroids di CPU
//       e. Cek konvergensi
//    4. Cleanup
// ============================================================
void kmeansOpenCL(const float* pixels, int N, int K,
                  float* centroids, int* labels,
                  int maxIter, const float* invCov,
                  const char* kernelPath) {

    cl_int err;

    // ---- 1. Platform & Device ----
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    checkCL(err, "clGetPlatformIDs");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        cout << "  [OpenCL] GPU tidak ditemukan, memakai CPU..." <<  endl;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        checkCL(err, "clGetDeviceIDs (CPU fallback)");
    }

    // Print info device
    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    cout << "  [OpenCL] Device: " << deviceName <<  endl;

    // ---- 2. Context & Command Queue ----
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkCL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkCL(err, "clCreateCommandQueue");

    // ---- 3. Kompilasi Kernel ----
    string src = readKernelSource(kernelPath);
    const char* srcPtr = src.c_str();
    size_t srcLen = src.size();

    cl_program program = clCreateProgramWithSource(context, 1, &srcPtr, &srcLen, &err);
    checkCL(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print build log jika gagal
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), NULL);
        cerr << "[OpenCL BUILD ERROR]\n" << buildLog.data() <<  endl;
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "assign_labels", &err);
    checkCL(err, "clCreateKernel");

    // ---- 4. Buat Buffer GPU ----
    size_t pixelBytes    = N * 3 * sizeof(float);
    size_t centroidBytes = K * 3 * sizeof(float);
    size_t invCovBytes   = 9 * sizeof(float);
    size_t labelBytes    = N * sizeof(int);

    cl_mem dPixels = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        pixelBytes, (void*)pixels, &err);
    checkCL(err, "buffer pixels");

    cl_mem dCentroids = clCreateBuffer(context,
        CL_MEM_READ_ONLY,
        centroidBytes, NULL, &err);
    checkCL(err, "buffer centroids");

    cl_mem dInvCov = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        invCovBytes, (void*)invCov, &err);
    checkCL(err, "buffer invCov");

    cl_mem dLabels = clCreateBuffer(context,
        CL_MEM_WRITE_ONLY,
        labelBytes, NULL, &err);
    checkCL(err, "buffer labels");

    // Set kernel arguments yang tidak berubah
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dPixels);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dCentroids);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dInvCov);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &dLabels);
    clSetKernelArg(kernel, 4, sizeof(int), &N);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    // ---- 5. Info Workgroup (sekali, sebelum loop) ----
    size_t localWorkSize = 0;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(size_t), &localWorkSize, NULL);

    size_t globalWorkSize = N;
    size_t numWorkGroups  = (globalWorkSize + localWorkSize - 1) / localWorkSize;

    cout << "  [OpenCL] Work-group size (local)      : " << localWorkSize << " work-item" << endl;
    cout << "  [OpenCL] Jumlah workgroup             : " << numWorkGroups << " workgroup" << endl;

    // ---- 6. Loop Iterasi K-Means ----
    float* prevCentroids = new float[K * 3];

    for (int iter = 0; iter < maxIter; iter++) {
         memcpy(prevCentroids, centroids, centroidBytes);

        // Upload centroids ke GPU
        err = clEnqueueWriteBuffer(queue, dCentroids, CL_TRUE,
            0, centroidBytes, centroids, 0, NULL, NULL);
        checkCL(err, "write centroids");

        // Jalankan kernel assignment
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
            &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        checkCL(err, "enqueue kernel");

        // Download labels dari GPU
        err = clEnqueueReadBuffer(queue, dLabels, CL_TRUE,
            0, labelBytes, labels, 0, NULL, NULL);
        checkCL(err, "read labels");

        // Update centroids di CPU
        float* sums  = new float[K * 3]();
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
        }

        delete[] sums;
        delete[] counts;

        // Cek konvergensi
        bool converged = true;
        for (int j = 0; j < K * 3; j++) {
            if ( fabs(centroids[j] - prevCentroids[j]) > 1e-6f) {
                converged = false;
                break;
            }
        }

        if (converged) break;
    }

    // ---- 6. Cleanup ----
    delete[] prevCentroids;

    clReleaseMemObject(dPixels);
    clReleaseMemObject(dCentroids);
    clReleaseMemObject(dInvCov);
    clReleaseMemObject(dLabels);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
