#include "cuda_runtime.h"

#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

using namespace std;


/**
 * Multiplying a n*m and m*w matrix together to get a n*w matrix
*/
__global__ void cudaMatrixMultiply(double* a, double* b, double* out, int n, int m, int w) {

  int ROW = blockIdx.y * blockDim.y + threadIdx.y;
  int COL = blockIdx.x * blockDim.x + threadIdx.x;

  if (ROW < n && COL < w) {
    double tmp = 0.0;
    for (int i = 0; i < m; i++) {
      tmp += a[m * ROW + i] * b[i * w + COL];
    }
    out[ROW * w + COL] = tmp;
  }
}

void matrixMultiplication(double* a, double* b, double* out, int n, int m, int w) {
  dim3 threadsPerBlock(w, n);
  dim3 blocksPerGrid(1, 1);

  if (n * w > 512) {
    threadsPerBlock.x = 512;
    threadsPerBlock.y = 512;
    blocksPerGrid.x = ceil(double(w) / double(threadsPerBlock.x));
    blocksPerGrid.y = ceil(double(n) / double(threadsPerBlock.y));
  }

  cudaMatrixMultiply << <blocksPerGrid, threadsPerBlock >> > (a, b, out, n, m, w);
}

int main() {

  // Multiplying a 3x3 matrix by 3x3

  int n = 700;
  int m = 550;
  int w = 700;

  double mat_a[n * m];
  double mat_b[m * w];
  double mat_out[n * w];

  for (int i = 0; i < n * m; i++) {
    mat_a[i] = sin(i);
  }

  for (int i = 0; i < m * w; i++) {
    mat_b[i] = cos(i);
  }

  double* cudaA;
  double* cudaB;
  double* cudaOut;

  cudaMalloc(&cudaA, sizeof(mat_a));
  cudaMalloc(&cudaB, sizeof(mat_b));
  cudaMalloc(&cudaOut, sizeof(double) * n * w);

  cudaMemcpy(cudaA, mat_a, sizeof(mat_a), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaB, mat_b, sizeof(mat_b), cudaMemcpyHostToDevice);

  matrixMultiplication(cudaA, cudaB, cudaOut, n, m, w);

  cudaMemcpy(mat_out, cudaOut, sizeof(mat_out), cudaMemcpyDeviceToHost);

  cudaFree(cudaA);
  cudaFree(cudaB);
  cudaFree(cudaOut);

  return 0;
}
