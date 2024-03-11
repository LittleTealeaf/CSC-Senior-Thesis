#include "cuda_runtime.h"

#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

__global__ void cudaFeedForward(double *inputs, double *weights, double *bias,
                                double *out, int observations, int input,
                                int output) {
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

__global__ void cudaBackPropagateErrorOutput(double *activation,
                                             double *expected, double *out,
                                             int observations) {
  // idx is the observation
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < observations) {
    out[idx] = 0.0;
    if (activation[idx] > 0.0) {
      out[idx] = expected[idx] - activation[idx];
    }
  }
}

void runCudaBackPropagateErrorOutput(double *cuda_activation, double *expected,
                                     double *out, int observations) {
  dim3 threadsPerBlock(observations);
  dim3 blocksPerGrid(1);

  if (observations > 512) {
    threadsPerBlock.x = 512;
    blocksPerGrid.x = ceil(double(observations) / double(threadsPerBlock.x));
  }

  cudaBackPropagateErrorOutput<<<blocksPerGrid, threadsPerBlock>>>(
      cuda_activation, expected, out, observations);
}

__global__ void cudaBackPropagateError(double *activation, double *weights,
                                       double *errors, double *out,
                                       int observations, int layer_nodes,
                                       int next_nodes) {
  int obs = blockIdx.x * blockDim.x + threadIdx.x;  // Observation
  int node = blockIdx.y * blockDim.y + threadIdx.y; // Node

  if (obs < observations && node < layer_nodes) {
    double sum = 0.0;
    if (activation[obs * layer_nodes + node] > 0.0) {
      for (int i = 0; i < next_nodes; i++) {
        double weight = weights[node * next_nodes + i];
        double error = errors[obs * next_nodes + i];
        sum += weight * error;
      }
    }
    out[obs * layer_nodes + node] = sum;
  }
}

class CudaTensor {
private:
  int rows;
  int cols;
  double *cuda;

public:
  CudaTensor() : rows(0), cols(0), cuda(nullptr) {}

  CudaTensor(int rows, int cols) : cols(cols), rows(rows) {
    cudaMalloc(&this->cuda, this->cols * this->rows * sizeof(double));
  }

  double *getCuda() { return this->cuda; }

  void setValues(double *values) {
    cudaMemcpy(this->cuda, values, this->cols * this->rows * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  void getValues(double *values) {
    cudaMemcpy(values, this->cuda, this->cols * this->rows * sizeof(double),
               cudaMemcpyDeviceToHost);
  }

  void free() { cudaFree(this->cuda); }
};

class NetworkLayer {
private:
  CudaTensor weights;
  CudaTensor bias;

public:
  int input;
  int output;
  NetworkLayer(int input, int output)
      : input(input), output(output), weights(input, output), bias(output, 1) {}

  void setWeights(double *values) { this->weights.setValues(values); }

  void setBias(double *bias) { this->bias.setValues(bias); }

  void feedForward(double *cuda_in, int observations, double *cuda_out) {
    dim3 threadsPerBlock(this->output, observations);
    dim3 blocksPerGrid(1, 1);

    if (observations * this->output > 512) {
      threadsPerBlock.x = 512;
      threadsPerBlock.y = 512;
      blocksPerGrid.x = ceil(double(this->output) / double(threadsPerBlock.x));
      blocksPerGrid.y = ceil(double(observations) / double(threadsPerBlock.y));
    }

    cudaFeedForward<<<blocksPerGrid, threadsPerBlock>>>(
        cuda_in, this->weights.getCuda(), this->bias.getCuda(), cuda_out,
        observations, this->input, this->output);
  }

  void free() {
    this->weights.free();
    this->bias.free();
  }
};

class Network {
private:
  int input;
  int layer_count;
  vector<NetworkLayer> layers;

public:
  Network(ifstream stream) {
    this->input = -1;

    string line;

    while (getline(stream, line)) {
      // Line is currently <in> <out>
      int index = line.find(' ');
      int input = stoi(line.substr(0, index));
      if (this->input == -1) {
        this->input = input;
      }
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
      layers.push_back(layer);
    }
    stream.close();
    layer_count = layers.size();
  }

  // RELU 1 if positive else negative

  // TODO: convert inputs to be cuda so we can copy to GPU outside of timed
  // sections
  void train(int observations, double *inputs, double *expected) {
    double *cuda_inputs;
    cudaMalloc(&cuda_inputs, sizeof(double) * observations * this->input);
    cudaMemcpy(cuda_inputs, inputs, sizeof(double) * observations * this->input,
               cudaMemcpyHostToDevice);

    double *cuda_expected;
    cudaMalloc(&cuda_expected, sizeof(double) * observations);
    cudaMemcpy(cuda_expected, expected, sizeof(double) * observations,
               cudaMemcpyHostToDevice);

    double *outputs[layer_count];

    cudaMalloc(&outputs[0],
               sizeof(double) * layers.at(0).output * observations);
    layers.at(0).feedForward(cuda_inputs, observations, outputs[0]);

    for (int i = 1; i < layer_count; i++) {
      cudaMalloc(&outputs[i],
                 sizeof(double) * layers.at(i).output * observations);
      layers.at(i).feedForward(outputs[i - 1], observations, outputs[i]);
    }

    double *errors[layer_count];

    for (int i = 0; i < layer_count; i++) {
      cudaMalloc(&errors[i],
                 sizeof(double) * observations * layers.at(i).output);
    }

		// TODO MAYBE REBUILD THIS CAUSE I DONT KNOW WHAT I'M DOING
    runCudaBackPropagateErrorOutput(outputs[layer_count - 1], cuda_expected,
                                    errors[layer_count - 1], observations);

    cudaFree(cuda_inputs);
    cudaFree(cuda_expected);
    for (int i = 0; i < layer_count; i++) {
      cudaFree(outputs[i]);
      cudaFree(errors[i]);
    }
  }

  ~Network() {
    for (NetworkLayer layer : this->layers) {
      layer.free();
    }
  }
};

int main() {
  Network network = Network(ifstream(getenv("NETWORK")));
  double input[] = {1.0, 0.0};
  double expected[] = {3.0};

  network.train(1, input, expected);
}
