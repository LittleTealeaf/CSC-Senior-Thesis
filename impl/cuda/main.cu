#include "cuda_runtime.h"

#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <vector>
#include <fstream>

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

  void free() {
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

  void free() {
    this->weights.free();
    this->bias.free();
  }
};

class Network {
private:
  vector<NetworkLayer> layers;
public:
  Network(ifstream stream) {

    string line;

    while (getline(stream, line)) {
      // Line is currently <in> <out>
      int index = line.find(' ');
      int input = stoi(line.substr(0, index));
      int output = stoi(line.substr(index + 1));

      // Next line is the bias
      getline(stream, line);
      double bias[output];
      for (int o = 0; o < output - 1; o++) {
        index = line.find(' ');
        bias[o] = stod(line.substr(0, index));
        line = line.substr(index + 1);
      }
      bias[output - 1] = stod(line);

      // Weights
      double weights[input * output];
      for (int i = 0; i < input; i++) {
        getline(stream, line);
        for (int o = 0; o < output - 1; o++) {
          index = line.find(' ');
          weights[i * output + o] = stod(line.substr(0, index));
          line = line.substr(index + 1);
        }
        weights[(i + 1) * output - 1] = stod(line);
      }
      getline(stream, line);

      // Now we pushit into a layer
      NetworkLayer layer = NetworkLayer(input, output);
      layer.setWeights(weights);
      layer.setBias(bias);
      this->layers.push_back(layer);
    }
    stream.close();
  }

  ~Network() {
    for (NetworkLayer layer : this->layers) {
      layer.free();
    }
  }
};




int main() {
  Network network = Network(ifstream(getenv("NETWORK")));
}
