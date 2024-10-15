#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace Net
{

class NeuralNetworkTrainer
{
  public:
    NeuralNetworkTrainer()
    {
        initializeWeights();
    }

    std::vector<std::vector<std::vector<double>>> getWeights() const
    {
        return weights;
    }

    void train(const std::vector<std::vector<bool>> &data, const std::vector<std::vector<double>> &targets, int epochs,
               double learningRate)
    {
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            for (size_t i = 0; i < data.size(); ++i)
            {
                std::vector<double> input(data[i].begin(), data[i].end());
                std::vector<double> hidden = forward(input, 0);
                std::vector<double> output = forward(hidden, 1);
                backward(input, hidden, output, targets[i], learningRate);
            }
        }
    }

  private:
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;

    void initializeWeights()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 1);

        weights.resize(2);
        weights[0].resize(10, std::vector<double>(1024));
        weights[1].resize(16, std::vector<double>(10));

        biases.resize(2);
        biases[0].resize(10, 0.0);
        biases[1].resize(16, 0.0);

        for (auto &layer : weights)
        {
            for (auto &neuron : layer)
            {
                double initFactor = sqrt(2.0 / neuron.size());
                for (double &weight : neuron)
                {
                    weight = d(gen) * initFactor;
                }
            }
        }
    }

    std::vector<double> forward(const std::vector<double> &input, int layer)
    {
        std::vector<double> output(weights[layer].size(), 0.0);

        for (size_t i = 0; i < output.size(); ++i)
        {
            for (size_t j = 0; j < input.size(); ++j)
            {
                output[i] += input[j] * weights[layer][i][j];
            }
            output[i] += biases[layer][i];
            output[i] = sigmoid(output[i]);
        }

        return output;
    }

    void backward(const std::vector<double> &input, const std::vector<double> &hidden,
                  const std::vector<double> &output, const std::vector<double> &target, double learningRate)
    {
        std::vector<double> outputError(output.size());
        std::vector<double> hiddenError(hidden.size());

        // Calculate output layer error
        for (size_t i = 0; i < output.size(); ++i)
        {
            outputError[i] = (output[i] - target[i]) * sigmoidDerivative(output[i]);
        }

        // Calculate hidden layer error
        for (size_t i = 0; i < hidden.size(); ++i)
        {
            hiddenError[i] = 0.0;
            for (size_t j = 0; j < output.size(); ++j)
            {
                hiddenError[i] += outputError[j] * weights[1][j][i];
            }
            hiddenError[i] *= sigmoidDerivative(hidden[i]);
        }

        // Update weights and biases for hidden to output
        for (size_t i = 0; i < output.size(); ++i)
        {
            for (size_t j = 0; j < hidden.size(); ++j)
            {
                weights[1][i][j] -= learningRate * outputError[i] * hidden[j];
            }
            biases[1][i] -= learningRate * outputError[i];
        }

        // Update weights and biases for input to hidden
        for (size_t i = 0; i < hidden.size(); ++i)
        {
            for (size_t j = 0; j < input.size(); ++j)
            {
                weights[0][i][j] -= learningRate * hiddenError[i] * input[j];
            }
            biases[0][i] -= learningRate * hiddenError[i];
        }
    }

    static double sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }

    static double sigmoidDerivative(double x)
    {
        return x * (1.0 - x);
    }
};

class NeuralNetworkGenerator
{
  public:
    NeuralNetworkGenerator(const std::vector<std::vector<std::vector<double>>> &weights) : weights(weights)
    {
    }

    std::vector<double> generate(const std::vector<bool> &input)
    {
        std::vector<double> inputDouble(input.begin(), input.end());
        std::vector<double> hidden = forward(inputDouble, 0);
        return forward(hidden, 1);
    }

  private:
    std::vector<std::vector<std::vector<double>>> weights;

    std::vector<double> forward(const std::vector<double> &input, int layer)
    {
        std::vector<double> output(weights[layer].size(), 0.0);

        for (size_t i = 0; i < output.size(); ++i)
        {
            for (size_t j = 0; j < input.size(); ++j)
            {
                output[i] += input[j] * weights[layer][i][j];
            }
            output[i] = sigmoid(output[i]);
        }

        return output;
    }

    static double sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }
};

inline void save(const NeuralNetworkTrainer &d, const char *filename)
{
    auto data = d.getWeights();
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        std::cerr << "Failed to open file for writing\n";
        return;
    }
    for (auto &layer : data)
    {
        for (auto &neuron : layer)
        {
            for (double weight : neuron)
            {
                fprintf(f, "%.12lf ", weight);
            }
        }
    }
    fclose(f);
}

inline NeuralNetworkGenerator read(const char *filename)
{
    std::vector<std::vector<std::vector<double>>> data(2);
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        std::cerr << "Failed to open file for reading\n";
        throw std::runtime_error("File open failed");
    }

    data[0].resize(10, std::vector<double>(1024));
    for (auto &neuron : data[0])
    {
        for (double &weight : neuron)
        {
            if (fscanf(f, "%lf ", &weight) != 1)
            {
                fclose(f);
                throw std::runtime_error("File read failed");
            }
        }
    }

    data[1].resize(16, std::vector<double>(10));
    for (auto &neuron : data[1])
    {
        for (double &weight : neuron)
        {
            if (fscanf(f, "%lf ", &weight) != 1)
            {
                fclose(f);
                throw std::runtime_error("File read failed");
            }
        }
    }

    fclose(f);
    return NeuralNetworkGenerator(data);
}

} // namespace Net