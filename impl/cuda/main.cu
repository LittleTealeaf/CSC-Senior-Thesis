#include "cuda_runtime.h"

#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

__global__ void cudaFeedForward(double *in, double *weights, double *output,
                                double *activations, int n, int f, int o) {
  // x <-- 0..n
  int ROW = blockIdx.x * blockDim.x + threadIdx.x;
  // y <-- 0..o
  int COL = blockIdx.y * blockDim.y + threadIdx.y;

  if (ROW < n && COL < o) {
    double total = weights[f * o + COL];
    for (int i = 0; i < f; i++) {
      total += in[ROW * n + i] * weights[i * o + COL];
    }
    int index = f * o * COL;
    output[index] = total;

    // Activation
    if (total <= 0.0) {
      total = 0.0;
    }
    activations[index] = total;
  }
}

__global__ void cudaBackPropagateOutput(double *a_j, double *in_o, double *a_o,
                                        double *exp, double *out_error,
                                        double *out_nudge, int n, int f) {
  // x <-- 0..n
  int ROW = blockIdx.x * blockDim.x + threadIdx.x;
  // y <-- 0..f
  int COL = blockIdx.y * blockDim.y + threadIdx.y;

  if (ROW < n && COL < f + 1) {
    double error = 0.0;
    if (a_o[ROW] > 0.0) {
      error = exp[ROW] - a_o[ROW];
    }

    if (COL == f) {
      out_error[ROW] = error;
      out_nudge[(ROW + 1) * (f + 1) - 1] = error;
    } else {
      out_nudge[ROW * (f + 1) + COL] = a_j[f * ROW + COL] * error;
    }
  }
}

__global__ void cudaBackPropagation(double *a_i, double *in_j, double *a_j,
                                    double *w_k, double *err_k,
                                    double *out_nudge, double *out_err_j, int N,
                                    int F, int O, int P) {

	// 0..N
	int n = blockIdx.x * blockDim.x + threadIdx.x;



  // // x <-- 0..n
  // int ROW = blockIdx.x * blockDim.x + threadIdx.x;
  // // y <-- 0..o
  // int COL = blockIdx.y * blockDim.y + threadIdx.y;
  //
  // if (ROW < N && COL < O + 1) {
  //
  //   if (COL == O) {
  //     // bias
  //   } else {
  //     double error = 0.0;
  //     // Derivative of in_j
  //     if (in_j[ROW * O + COL] > 0.0) {
  //       for (int k = 0; k < P; k++) {
  //         error += w_k[COL * P + k] * err_k[ROW * O + k];
  //       }
  //     }
  //
		// 	out_err_j[ROW * O + COL]
  //
  //   }
  // }
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

    // cudaFeedForward << <blocksPerGrid, threadsPerBlock >> > (
    //   cuda_in, this->weights.getCuda(), this->bias.getCuda(), cuda_out,
    //   observations, this->input, this->output);
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

  // TODO: convert inputs to be cuda so we can copy to GPU outside of timed
  // sections
  void train(int observations, double *cuda_inputs, double *cuda_expected) {

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

    // // TODO MAYBE REBUILD THIS CAUSE I DONT KNOW WHAT I'M DOING
    // runCudaBackPropagateErrorOutput(outputs[layer_count - 1], cuda_expected,
    //                                 errors[layer_count - 1], observations);

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

    double *cuda_inputs;
    cudaMalloc(&cuda_inputs, sizeof(double) * bootstrap.size() * features);
    cudaMemcpy(cuda_inputs, train_data,
               sizeof(double) * bootstrap.size() * features,
               cudaMemcpyHostToDevice);

    double *cuda_expected;
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
