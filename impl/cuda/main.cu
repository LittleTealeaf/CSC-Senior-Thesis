#include "cuda_runtime.h"

#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;



/// @brief Run Feed Forward progress from one layer to the next
/// @param in (N x F): The input values
/// @param weights (F x O) + (O x 1): Both the weights (first F x O) and bias (last O x 1)
/// @param output (N x O): Outputs of the network (before activation function)
/// @param activations (N x O): Activations of the nodes
/// @param N number of observations
/// @param F number of incoming features
/// @param O number of output nodes
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

/// @brief Back Propagate for the Output
/// @param a_j (N x 1): Activations of the previous layer
/// @param in_o (N x 1): Inputs for output layer
/// @param a_o (N x 1): Activations for output layer
/// @param exp (N x 1): Expected values for the output
/// @param out_error (N x F): Error for the previous nodes
/// @param out_nudge (F x 1) (1 x 1): Nudges for the weights to the output node
/// @param N number of observations
/// @param F number of features from the previous layer
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

/// @brief Cuda Back Propagation for layers that are not the output layer
/// @param a_i (N x F): Activations for previous layer
/// @param in_j (N x O): Inputs for current layer
/// @param a_j (N x O): Activation for current layer
/// @param w_k (O x P) (P x 1): Weights/Bias for next layer
/// @param err_k (N x P): Errors for next layer
/// @param out_nudge (N x ((F x O) + (O x 1))): Nudges for this layer
/// @param out_err_j (N x O): Errors for this layer
/// @param N number of observations
/// @param F features of previous layer
/// @param O output of current layer
/// @param P output of next layer
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

/// @brief Applies nudges. Note that dim.x needs to range from [0-F]
/// @param nudges (N x ((F x O) + (O x 1)))
/// @param weights (F x O) + (O x 1)
/// @param N
/// @param F
/// @param O
__global__ void cudaApplyNudges(double* nudges, double* weights, int N, int F, int O, double learning_rate) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  int o = blockIdx.y * blockDim.y + threadIdx.y;

  if (f <= F && o < O) {
    double nudge = 0.0;
    for (int n = 0; n < N; n++) {
      nudge += nudges[n * (F + 1) * O + f * O + o];
    }
    nudge *= 0.1 / double(N);
    weights[f * O + o] += nudge;
  }
}

class CudaGrid {
public:
  dim3 threads;
  dim3 blocks;

  CudaGrid(int x) : CudaGrid(x, 1, 1) {}

  CudaGrid(int x, int y) : CudaGrid(x, y, 1) {}

  CudaGrid(int x, int y, int z) : threads(1, 1, 1), blocks(1, 1, 1) {
    setX(x);
    setY(y);
    setZ(z);
  }

  void setX(int x) {
    blocks.x = ceil(double(x) / 8.0);
    threads.x = min(8, x);
  }

  void setY(int y) {
    blocks.y = ceil(double(y) / 8.0);
    threads.y = min(8, y);
  }

  void setZ(int z) {
    blocks.z = ceil(double(z) / 8.0);
    threads.z = min(8, z);
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
public:
  int input;
  int output;
  double* cuda;

  NetworkLayer(int input, int output)
    : input(input), output(output), cuda(nullptr) {
    cudaMalloc(&this->cuda, (input + 1) * output * sizeof(double));
  }

  void setValues(double* values) {
    cudaMemcpy(this->cuda, values, (this->input + 1) * this->output * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  void free() {
    cudaFree(this->cuda);
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
      double weights[(input + 1) * output];
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

      for (int i = 0; i < output; i++) {
        weights[input * output + i] = bias[i];
      }

      // Now we pushit into a layer
      NetworkLayer layer = NetworkLayer(input, output);
      layer.setValues(weights);
      layers.push_back(layer);
    }
    stream.close();
    layer_count = layers.size();
  }

  // RELU 1 if positive else negative


  void train(int observations, double* cuda_inputs, double* cuda_expected) {
    double* outputs[layer_count];
    double* activations[layer_count];

    {
      int size = sizeof(double) * layers.at(0).output * observations;
      cudaMalloc(&outputs[0], size);
      cudaMalloc(&activations[0], size);
    }

    CudaGrid grid = CudaGrid(observations, layers.at(0).input);

    // Feed Forward First Layer

    {
      NetworkLayer* layer = &layers.at(0);
      cudaFeedForward << <grid.blocks, grid.threads >> > (cuda_inputs, layer->cuda, outputs[0], activations[0], observations, layer->input, layer->output);
    }

    // Feed Forward Rest

    for (int i = 1; i < layer_count; i++) {
      NetworkLayer* layer = &layers.at(i);
      int size = sizeof(double) * layer->output * observations;
      cudaMalloc(&outputs[i], size);
      cudaMalloc(&activations[i], size);
      grid.setY(layer->output);

      cudaFeedForward << <grid.blocks, grid.threads >> > (activations[i - 1], layer->cuda, outputs[i], activations[i], observations, layer->input, layer->output);

    }

    double* errors[layer_count];
    double* nudges[layer_count];

    for (int i = 0; i < layer_count; i++) {
      NetworkLayer* layer = &layers.at(i);
      cudaMalloc(&errors[i], sizeof(double) * observations * layer->output);
      cudaMalloc(&nudges[i], sizeof(double) * observations * layer->output * (layer->input + 1));
    }


    // Back Prop Layer Layer
    {
      int prev_features = layers.at(layer_count - 2).output;
      grid.setY(prev_features);
      cudaBackPropagateOutput << <grid.blocks, grid.threads >> > (activations[layer_count - 2], outputs[layer_count - 1], activations[layer_count - 1], cuda_expected, errors[layer_count - 1], nudges[layer_count - 1], observations, prev_features);
    }

    // Middle Layers

    for (int i = layer_count - 2; i > 0; i--) {
      NetworkLayer* layer = &layers.at(i);
      grid.setY(layer->input);
      cudaBackPropagation << <grid.blocks, grid.threads >> > (activations[i - 1], outputs[i], activations[i], layers.at(i + 1).cuda, errors[i + 1], nudges[i], errors[i], observations, layer->input, layer->output, layers.at(i + 1).output);
    }

    // First Layer

    {
      NetworkLayer* layer = &layers.at(0);
      grid.setY(layer->input);
      cudaBackPropagation << <grid.blocks, grid.threads >> > (cuda_inputs, outputs[0], activations[0], layers.at(1).cuda, errors[1], nudges[0], errors[0], observations, layer->input, layer->output, layers.at(1).output);
    }

    // Apply Nudges
    for (int i = 0; i < layer_count; i++) {
      NetworkLayer* layer = &layers.at(i);
      grid.setX(layer->input);
      grid.setY(layer->output + 1);
      cudaApplyNudges << <grid.blocks, grid.threads >> > (nudges[i], layer->cuda, observations, layer->input, layer->output, 0.1);
    }


    for (int i = 0;i < layer_count; i++) {
      cudaFree(outputs[i]);
      cudaFree(activations[i]);
      cudaFree(errors[i]);
      cudaFree(nudges[i]);
    }
  }

  ~Network() {
    for (NetworkLayer layer : this->layers) {
      layer.free();
    }
  }
};

int main() {

  Network network = Network(ifstream(getenv("NETWORK_PATH")));
  int features = network.features;

  vector<double> data;

  // Read data file
  ifstream stream(getenv("DATASET_PATH"));
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
  stream = ifstream(getenv("BOOTSTRAP_PATH"));
  while (getline(stream, line)) {
    vector<int> choices;
    int index;
    int count = stoi(getenv("BOOTSTRAP_COUNT"));
    for(int i = 0; i < count; i++) {
      if ((index = line.find(',')) >= 0) {
        choices.push_back(stoi(line.substr(0,index)));
        line =line.substr(index + 1);
      } else {
        choices.push_back(stoi(line));
      }
    }

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

  ofstream out_file(getenv("OUT_PATH"));
  if (out_file.is_open()) {
    for (int i = 0; i < times.size(); ++i) {
      out_file << i << "," << times.at(i) << "\n";
    }

    out_file.close();
  }

  data.clear();
  bootstraps.clear();
}
