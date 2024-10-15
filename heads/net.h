#include <cmath>
#include <iostream>
#include <random>
#include <vector>

class NeuralNetworkTrainer
{
  public:
    NeuralNetworkTrainer()
    {
        // Initialize weights with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for (auto &layer : weights)
        {
            for (auto &row : layer)
            {
                for (auto &weight : row)
                {
                    weight = dis(gen);
                }
            }
        }
    }

    void train(const std::vector<bool> &input, const std::vector<double> &target)
    {
        // Forward propagation
        std::vector<double> hidden(10, 0.0); // 10 hidden neurons
        std::vector<double> output(16, 0.0);

        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < 1024; ++j)
            {
                hidden[i] += weights[0][i][j] * input[j];
            }
            hidden[i] = sigmoid(hidden[i]);
        }

        for (int i = 0; i < 16; ++i)
        {
            for (int j = 0; j < 10; ++j)
            {
                output[i] += weights[1][i][j] * hidden[j];
            }
            output[i] = sigmoid(output[i]);
        }

        // Backward propagation
        std::vector<double> output_error(16, 0.0);
        std::vector<double> hidden_error(10, 0.0);

        for (int i = 0; i < 16; ++i)
        {
            output_error[i] = (target[i] - output[i]) * sigmoid_derivative(output[i]);
        }

        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < 16; ++j)
            {
                hidden_error[i] += output_error[j] * weights[1][j][i];
            }
            hidden_error[i] *= sigmoid_derivative(hidden[i]);
        }

        // Update weights
        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < 1024; ++j)
            {
                weights[0][i][j] += learning_rate * hidden_error[i] * input[j];
            }
        }

        for (int i = 0; i < 16; ++i)
        {
            for (int j = 0; j < 10; ++j)
            {
                weights[1][i][j] += learning_rate * output_error[i] * hidden[j];
            }
        }
    }

    std::vector<std::vector<std::vector<double>>> getWeights() const
    {
        return weights;
    }

  private:
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoid_derivative(double x)
    {
        return x * (1.0 - x);
    }

    std::vector<std::vector<std::vector<double>>> weights = {
        std::vector<std::vector<double>>(10, std::vector<double>(1024, 0.0)),
        std::vector<std::vector<double>>(16, std::vector<double>(10, 0.0))};

    double learning_rate = 0.1;
};

class NeuralNetworkGenerator
{
  public:
    NeuralNetworkGenerator(const std::vector<std::vector<std::vector<double>>> &trained_weights)
        : weights(trained_weights)
    {
    }

    std::vector<double> generate(const std::vector<bool> &input)
    {
        std::vector<double> hidden(10, 0.0);
        std::vector<double> output(16, 0.0);

        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < 1024; ++j)
            {
                hidden[i] += weights[0][i][j] * input[j];
            }
            hidden[i] = sigmoid(hidden[i]);
        }

        for (int i = 0; i < 16; ++i)
        {
            for (int j = 0; j < 10; ++j)
            {
                output[i] += weights[1][i][j] * hidden[j];
            }
            output[i] = sigmoid(output[i]);
        }

        return output;
    }

  private:
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    std::vector<std::vector<std::vector<double>>> weights;
};
inline void save(const NeuralNetworkTrainer &d)
{
    auto data = d.getWeights();
    FILE *f = fopen("./data.txt", "w");
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 1024; j++)
        {
            fprintf(f, "%.5lf ", data[0][i][j]);
        }
    }
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            fprintf(f, "%.5lf ", data[1][i][j]);
        }
    }
    fclose(f);
}
inline NeuralNetworkGenerator read()
{
    std::vector<std::vector<std::vector<double>>> data(2);
    FILE *f = fopen("./data.txt", "r");
    data[0].resize(10);
    for (int i = 0; i < 10; i++)
    {
        data[0][i].resize(1024);
        for (int j = 0; j < 1024; j++)
        {
            fscanf(f, "%lf ", &data[0][i][j]);
        }
    }
    data[1].resize(16);
    for (int i = 0; i < 16; i++)
    {
        data[1][i].resize(10);
        for (int j = 0; j < 10; j++)
        {
            fscanf(f, "%lf ", &data[1][i][j]);
        }
    }
    fclose(f);
    return NeuralNetworkGenerator(data);
}