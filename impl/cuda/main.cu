#include "cuda_runtime.h"

#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

using namespace std;



__global__ void cudaFeedForward(double* inputs, double* weights, double* bias, double* out, int observations, int input, int output) {
  int COL = blockIdx.x * blockDim.x + threadIdx.x; // Ouput
  int ROW = blockIdx.y * blockDim.y + threadIdx.y; //  Observation

  if (ROW < observations && COL < output) {
    double value = 0.0;
    for (int i = 0; i < input; i++) {
      value += inputs[input * ROW + i] * weights[i * output + COL];
    }
    value += bias[COL];
    out[ROW * output + COL] = max(value, 0.0);
  }
}

/**
 * Multiplying a n*m and m*w matrix together to get a n*w matrix
 */
__global__ void cudaMatrixMultiply(double* a, double* b, double* out, int n, int m, int w)
{

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


class CudaTensor {
private:
  int rows;
  int cols;
  double* cuda;

public:
  CudaTensor() : rows(0), cols(0), cuda(nullptr) {}

  CudaTensor(int rows, int cols) : cols(cols), rows(rows) {
    cudaMalloc(&this->cuda, this->cols * this->rows * sizeof(double));
  }

  double* getCuda() {
    return this->cuda;
  }

  void setValues(double* values) {
    cudaMemcpy(this->cuda, values, this->cols * this->rows * sizeof(double), cudaMemcpyHostToDevice);
  }

  void getValues(double* values) {
    cudaMemcpy(values, this->cuda, this->cols * this->rows * sizeof(double), cudaMemcpyDeviceToHost);
  }

  ~CudaTensor() {
    cudaFree(this->cuda);
  }
};



class NetworkLayer {
private:
  int input;
  int output;
  CudaTensor weights;
  CudaTensor bias;
public:
  NetworkLayer(int input, int output) : input(input), output(output), weights(input, output), bias(output, 1) {}


  void setWeights(double* values) {
    this->weights.setValues(values);
  }

  void setBias(double* bias) {
    this->bias.setValues(bias);
  }

  void feedForward(double* cuda_in, int observations, double* cuda_out) {
    dim3 threadsPerBlock(this->output, observations);
    dim3 blocksPerGrid(1, 1);

    if (observations * this->output > 512) {
      threadsPerBlock.x = 512;
      threadsPerBlock.y = 512;
      blocksPerGrid.x = ceil(double(this->output) / double(threadsPerBlock.x));
      blocksPerGrid.y = ceil(double(observations) / double(threadsPerBlock.y));
    }

    cudaFeedForward << <blocksPerGrid, threadsPerBlock >> > (cuda_in, this->weights.getCuda(), this->bias.getCuda(), cuda_out, observations, this->input, this->output);
  }
};



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
  NetworkLayer layer = NetworkLayer(2, 2);
  double weights[] = { 1.0,2.0,3.0,4.0 };
  double bias[] = { 2.0,1.0 };
  layer.setWeights(weights);
  layer.setBias(bias);

  double input[] = { 1.0,1.0,2.0,2.0 };

  double* cuda_input;
  cudaMalloc(&cuda_input, sizeof(input));
  cudaMemcpy(cuda_input, input, sizeof(input), cudaMemcpyHostToDevice);

  double* cuda_output;
  cudaMalloc(&cuda_output, sizeof(double) * 4);

  layer.feedForward(cuda_input, 2, cuda_output);


  double output[4];
  cudaMemcpy(output, cuda_output, sizeof(output), cudaMemcpyDeviceToHost);


  cout << output[0] << " " << output[1] << " " << output[2] << " " << output[3] << endl;

  cudaFree(cuda_input);
  cudaFree(cuda_output);
}
