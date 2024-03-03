#include "cuda_runtime.h"

#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

using namespace std;

__global__ void cudaMatrixMultiply(double *vec, double *mat, double *res) {}

int main() {

  double vec[] = {0.1, 0.2, 0.3};

  double mat[5 * 3];
  for (int i = 0; i < 5 * 3; i++) {
    mat[i] = 0.1 * i;
  }

  double *cudaVec = 0;
  double *cudaMat = 0;
  double *cudaRes = 0;

  cudaMalloc(&cudaVec, sizeof(vec));
  cudaMalloc(&cudaMat, sizeof(mat));
  cudaMalloc(&cudaRes, sizeof(double) * 3 * 5 * 3);

  cudaMemcpy(cudaVec, vec, sizeof(vec), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaMat, mat, sizeof(mat), cudaMemcpyHostToDevice);

  cudaFree(&cudaVec);
  cudaFree(&cudaMat);
  cudaFree(&cudaRes);

  return 0;
}
