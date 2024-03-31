#include "cuda_runtime.h"

#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

__global__ void cudaFeedForward(double* in, double* weights, double* output,
                                double* activations, int N, int F, int O) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int o = blockIdx.y * blockDim.y + threadIdx.y;

  if (n < N && o < O) {
    double total = weights[F * O + o];
    for (int i = 0; i < F; i++) {
      total += in[n * N + i] * weights[i * O + o];
    }
    int index = F * O * o;
    output[index] = total;

    // Activation
    if (total <= 0.0) {
      total = 0.0;
    }
    activations[index] = total;
  }
}

__global__ void cudaBackPropagateOutput(double* a_j, double* in_o, double* a_o,
                                        double* exp, double* out_error,
                                        double* out_nudge, int N, int F) {
  // 0..N
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  // 0..F
  int f = blockIdx.y * blockDim.y + threadIdx.y;

  if (n < N && f < F + 1) {
    double error = 0.0;
    if (a_o[n] > 0.0) {
      error = exp[n] - a_o[n];
    }

    if (f == F) {
      out_error[n] = error;
      out_nudge[(n + 1) * (F + 1) - 1] = error;
    }
    else {
      out_nudge[n * (F + 1) + f] = a_j[F * n + f] * error;
    }
  }
}

__global__ void cudaBackPropagation(double* a_i, double* in_j, double* a_j,
                                    double* w_k, double* err_k,
                                    double* out_nudge, double* out_err_j, int N,
                                    int F, int O, int P) {

  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int o = blockIdx.y * blockDim.y + threadIdx.y;

  if (n < N && o < O) {
    out_err_j[n * O + o] = 0.0;
    if (in_j[n * O + o] > 0.0) {
      for (int p = 0; p < P; p++) {
        out_err_j[n * O + o] += w_k[o * O + p] * err_k[n * P + p];
      }
    }

    for (int f = 0; f < F + 1; f++) {
      if (f == F) {
        out_nudge[n * ((F + 1) * O) + f * O + o] = out_err_j[n * O + o];
      }
      else {
        out_nudge[n * ((F + 1) * O) + f * O + o] = a_i[n * F + f] * out_err_j[n * O + o];
      }
    }
  }
}

class CudaGrid {
public:
  dim3 threads;
  dim3 blocks;

  CudaGrid(int x) : CudaGrid(x, 1, 1) {}

  CudaGrid(int x, int y) : CudaGrid(x, y, 1) {}

  CudaGrid(int x, int y, int z) : threads(x, y, z), blocks(x, y, z) {
    if (threads.x > 8) {
      blocks.x = ceil(double(threads.x) / double(8));
      threads.x = 8;
    }
    if (threads.y > 8) {
      blocks.y = ceil(double(threads.y) / double(8));
      threads.y = 8;
    }
    if (threads.z > 8) {
      blocks.z = ceil(double(threads.z) / double(8));
      threads.z = 8;
    }
  }
};


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

  double* getCuda() { return this->cuda; }

  void setValues(double* values) {
    cudaMemcpy(this->cuda, values, this->cols * this->rows * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  void getValues(double* values) {
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
    : input(input), output(output), weights(input, output), bias(output, 1) {
  }

  void setWeights(double* values) { this->weights.setValues(values); }

  void setBias(double* bias) { this->bias.setValues(bias); }

  void feedForward(double* cuda_in, int observations, double* cuda_out) {

    CudaGrid grid = CudaGrid(this->output, observations);
    cudaFeedForward << <grid.blocks, grid.threads >> > (cuda_in, this->weights.getCuda(), this->bias.getCuda(), cuda_out, observations, this->input, this->output);
  }

  void free() {
    this->weights.free();
    this->bias.free();
  }
};

class Network {
private:
  int layer_count;
  vector<NetworkLayer> layers;

public:
  int features;
  Network(ifstream stream) {
    this->features = -1;

    string line;

    while (getline(stream, line)) {
      // Line is currently <in> <out>
      int index = line.find(' ');
      int input = stoi(line.substr(0, index));
      if (this->features == -1) {
        this->features = input;
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
          index = line.find(',');
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


  void train(int observations, double* cuda_inputs, double* cuda_expected) {

    double* outputs[layer_count];

    cudaMalloc(&outputs[0],
               sizeof(double) * layers.at(0).output * observations);
    layers.at(0).feedForward(cuda_inputs, observations, outputs[0]);

    for (int i = 1; i < layer_count; i++) {
      cudaMalloc(&outputs[i],
                 sizeof(double) * layers.at(i).output * observations);
      layers.at(i).feedForward(outputs[i - 1], observations, outputs[i]);
    }

    double* errors[layer_count];

    for (int i = 0; i < layer_count; i++) {
      cudaMalloc(&errors[i],
                 sizeof(double) * observations * layers.at(i).output);
    }



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
  int features = network.features;

  vector<double> data;

  // Read data file
  ifstream stream(getenv("DATA"));
  string line;
  while (getline(stream, line)) {

    int index;
    for (int i = 0; i < network.features; i++) {
      index = line.find(',');
      data.push_back(stod(line.substr(0, index)));
      line = line.substr(index + 1);
    }
    data.push_back(stod(line));
  }
  stream.close();

  vector<vector<int>> bootstraps;
  stream = ifstream(getenv("BOOTSTRAP"));
  while (getline(stream, line)) {
    vector<int> choices;
    int index;
    while ((index = line.find(',')) >= 0) {
      choices.push_back(stoi(line.substr(0, index)));
      line = line.substr(index + 1);
    }
    choices.push_back(stoi(line));

    bootstraps.push_back(choices);
  }

  vector<std::chrono::nanoseconds::rep> times;

  for (vector<int> bootstrap : bootstraps) {
    double train_data[network.features * bootstrap.size()];
    double train_expected[bootstrap.size()];

    for (int i = 0; i < bootstrap.size(); i++) {
      int index = bootstrap.at(i);
      train_expected[i] = data.at((features + 1) * index);
      for (int j = 0; j < features; j++) {
        train_data[i + j] = data.at((features + 1) * index + 1 + j);
      }
    }

    double* cuda_inputs;
    cudaMalloc(&cuda_inputs, sizeof(double) * bootstrap.size() * features);
    cudaMemcpy(cuda_inputs, train_data,
               sizeof(double) * bootstrap.size() * features,
               cudaMemcpyHostToDevice);

    double* cuda_expected;
    cudaMalloc(&cuda_expected, sizeof(double) * bootstrap.size());
    cudaMemcpy(cuda_expected, train_expected, sizeof(double) * bootstrap.size(),
               cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    network.train(bootstrap.size(), cuda_inputs, cuda_expected);

    auto end = std::chrono::high_resolution_clock::now();

    times.push_back(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
      .count());
  }

  ofstream out_file(getenv("OUT_TIMES"));
  if (out_file.is_open()) {
    out_file << "id,time\n";
    for (int i = 0; i < times.size(); ++i) {
      out_file << i << "," << times.at(i) << "\n";
    }

    out_file.close();
  }

  data.clear();
  bootstraps.clear();
}
