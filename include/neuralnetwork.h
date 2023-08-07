
#pragma once
#include <vector>
#include "dataset.h"
#include "matrix.h"
namespace nn
{
    class NeuralNetwork
    {
    private:
        std::vector<size_t> arch;
        std::vector<Matrix> biases;
        std::vector<Matrix> weights;
        const double eps = 1e-3;
        const double rate = 1e-1;

    public:
        NeuralNetwork(std::vector<size_t> arch);

        void rand();
        Matrix forward(Matrix input);
        T cost(DataSet &train);
        void finiteDiff(DataSet &train);

        void printArch();
        void printWeights();
        void printBiases();

        std::vector<size_t> getArch();
        T getBias(size_t layer, size_t nr);
        T getWeight(size_t layer1, size_t node1, size_t node2);
        T getRate();
    };

    NeuralNetwork::NeuralNetwork(std::vector<size_t> arch) : arch(arch)
    {
        weights.resize(arch.size() - 1);
        biases.resize(arch.size() - 1);
        for (size_t i = 0; i < arch.size() - 1; i++)
            weights[i] = nn::Matrix{arch[i], arch[i + 1]};

        for (size_t i = 0; i < arch.size() - 1; i++)
            biases[i] = Matrix{1, arch[i + 1]};
    }
   
    void NeuralNetwork::rand()
    {
        for (auto &w : weights)
            w.rand();

        for (auto &b : biases)
            b.rand();
    }
    Matrix NeuralNetwork::forward(Matrix input)
    {

        for (size_t i = 0; i < weights.size(); i++)
        {
            // forward the input add activate using sigmoid function
            input = (input * weights[i] + biases[i]).activate([](auto x)
                                                              { return 1.0 / (1.0 + exp(-x)); });
        }
        return input;
    }
    T NeuralNetwork::cost(DataSet &train)
    {
        T cost = 0;
        for (size_t i = 0; i < train.size(); i++)
        {
            // actual value
            Matrix act = forward(train.getInputMat(i));

            // expected value
            Matrix exp{1, train.getData(i).output.size()};
            exp = train.getOutputMat(i);

            assert(act.shape == exp.shape);
            for (size_t j = 0; j < act.data.size(); j++)
            {
                T d = act.data[j] - exp.data[j];
                cost += d * d;
            }
        }
        return cost / static_cast<T>(train.size());
    }
    void NeuralNetwork::finiteDiff(DataSet &train)
    {
        auto c = cost(train);

        std::vector<Matrix> W = weights;
        for (size_t i = 0; i < weights.size(); i++)
            for (size_t j = 0; j < weights[i].data.size(); j++)
            {
                weights[i].data[j] += eps;
                // (f(x + eps) - f(x)) / eps
                double dw = (cost(train) - c) / eps;

                // reset it back
                weights[i].data[j] -= eps;

                W[i].data[j] -= rate * dw;
            }

        std::vector<Matrix> B = biases;
        for (size_t i = 0; i < biases.size(); i++)
            for (size_t j = 0; j < biases[i].data.size(); j++)
            {
                biases[i].data[j] += eps;
                // (f(x + eps) - f(x)) / eps
                double db = (cost(train) - c) / eps;

                // reset it back
                biases[i].data[j] -= eps;

                B[i].data[j] -= rate * db;
            }
        weights = W;
        biases = B;
    }
  
    void NeuralNetwork::printArch()
    {
        std::cout << "NeuralNetwork Arch([";
        for (size_t i = 0; i < arch.size() - 1; i++)
            std::cout << arch[i] << ',';
        std::cout << arch.back() << "])\n";
    }
    void NeuralNetwork::printWeights()
    {
        std::cout << "NeuralNetwork = [\n";
        for (size_t i = 0; i < weights.size(); i++)
        {
            std::cout << "  ";
            weights[i].printX("w" + std::to_string(i));
        }
        std::cout << "]\n";
    }
    void NeuralNetwork::printBiases()
    {
        std::cout << "NeuralNetwork = [\n";
        for (size_t i = 0; i < biases.size(); i++)
        {
            std::cout << "  ";
            biases[i].printX("b" + std::to_string(i));
        }
        std::cout << "]\n";
    }
  
    std::vector<size_t> NeuralNetwork::getArch()
    {
        return arch;
    }
    T NeuralNetwork::getBias(size_t layer, size_t nr)
    {
        return biases[layer](0, nr);
    }
    T NeuralNetwork::getWeight(size_t layer1, size_t node1, size_t node2)
    {
        return weights[layer1](node1, node2);
    }
    T NeuralNetwork::getRate()
    {
        return rate;
    }
}