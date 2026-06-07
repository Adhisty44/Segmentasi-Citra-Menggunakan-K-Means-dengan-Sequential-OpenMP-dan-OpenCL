# ============================================================
#  Makefile — K-Means Benchmark (Sequential + OpenMP + OpenCL)
#
#  Penggunaan:
#    make              → build program
#    make run          → build + jalankan (default: sawah6.png 100)
#    make clean        → hapus file hasil kompilasi
#
#  Sesuaikan path OpenCV dan OpenCL di bawah jika perlu.
# ============================================================

CXX       = g++
CXXFLAGS  = -O2 -std=c++17 -fopenmp -Wall

# --- OpenCV (sesuaikan jika perlu) ---
# Jika pakai pkg-config:
OPENCV_CFLAGS  = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS    = $(shell pkg-config --libs opencv4)

# Jika pkg-config tidak tersedia, uncomment dan sesuaikan:
# OPENCV_CFLAGS = -IC:/opencv/build/include
# OPENCV_LIBS   = -LC:/opencv/build/x64/vc16/lib -lopencv_world490

# --- OpenCL ---
# Jika OpenCL SDK terinstall secara sistem:
OPENCL_LIBS = -lOpenCL
# Jika perlu path khusus, uncomment dan sesuaikan:
# OPENCL_CFLAGS = -IC:/path/to/OpenCL/include
# OPENCL_LIBS   = -LC:/path/to/OpenCL/lib -lOpenCL

# --- Target ---
TARGET    = kmeans_benchmark
SRCS      = main.cpp utils.cpp sequential.cpp openmp_kmeans.cpp opencl_kmeans.cpp
OBJS      = $(SRCS:.cpp=.o)

# --- Default gambar (sesuaikan) ---
IMAGE     = sawah6.png
SIZE      = 100

# ============================================================
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS) $(OPENCL_LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENCL_CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(IMAGE) $(SIZE)

clean:
	rm -f $(OBJS) $(TARGET) result_*.png

.PHONY: all run clean
