#include "cuda_runtime.h"

#include <chrono>
#include <iostream>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

using namespace std;

__global__ void vectorAdd(int* a, int* b, int* c) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

int main() {

  for (int i = 0; i < 100; i++) {

    int a[] = { 1, 2, 3 };
    int b[] = { 4, 5, 6 };

    int c[sizeof(a) / sizeof(int)] = { 0 };

    int* cudaA = 0;
    int* cudaB = 0;
    int* cudaC = 0;

    cudaMalloc(&cudaA, sizeof(a));
    cudaMalloc(&cudaB, sizeof(b));
    cudaMalloc(&cudaC, sizeof(c));

    cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    vectorAdd << <1, sizeof(a) / sizeof(int) >> > (cudaA, cudaB, cudaC);

    auto finish = std::chrono::high_resolution_clock::now();

    cudaMemcpy(c, cudaC, sizeof(c), cudaMemcpyDeviceToHost);

    cudaFree(cudaA);
    cudaFree(cudaB);
    cudaFree(cudaC);

    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start)
      .count()
      << endl;
  }

  return 0;
}
