
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
        std::vector<Matrix> activations;

    public:
        double rate = 1;
        double eps = 1e-3;
        NeuralNetwork(std::vector<size_t> arch);
        void rand();
        Matrix forward(Matrix input);
        T cost(DataSet &train);

        void backProp(DataSet &train);
        void finiteDiff(DataSet &train);

        void printArch();
        void printWeights();
        void printBiases();

        std::vector<size_t> getArch();
        T getBias(size_t layer, size_t nr);
        T getWeight(size_t layer1, size_t node1, size_t node2);
    };
    NeuralNetwork::NeuralNetwork(std::vector<size_t> arch) : arch(arch)
    {
        weights.resize(arch.size() - 1);
        biases.resize(arch.size() - 1);
        activations.resize(arch.size());
        for (size_t i = 0; i < arch.size() - 1; i++)
            weights[i] = Matrix{arch[i], arch[i + 1]};

        for (size_t i = 0; i < arch.size() - 1; i++)
            biases[i] = Matrix{1, arch[i + 1]};

        for (size_t i = 0; i < arch.size(); i++)
            activations[i] = Matrix{1, arch[i]};
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
            activations[i] = input;
            input = (input * weights[i] + biases[i]).activate([](auto x)
                                                                       { return 1.0 / (1.0 + exp(-x)); });
            // Sussy behaviour if I replace input with activations
        }
        activations.back() = input;

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

            for (size_t j = 0; j < act.data.size(); j++)
            {
                T d = act(0, j) - exp(0, j);
                cost += d * d;
            }
        }
        return cost / static_cast<T>(train.size());
    }
    void NeuralNetwork::backProp(DataSet &train)
    {
        nn::NeuralNetwork g(this->arch);

        // i = current sample
        // l = current layer
        // j = current activation
        // k = previous activation
        size_t n = train.size();
        for (size_t i = 0; i < n; i++)
        {
            // expected value
            Matrix exp{1, train.getData(i).output.size()};
            exp = train.getOutputMat(i);
            forward(train.getInputMat(i));

            // reset the activations
            for (auto &a : g.activations)
                a.fill(0);

            for (size_t j = 0; j < exp.data.size(); j++)
                g.activations.back()(0, j) = activations.back()(0, j) - exp(0, j);

            for (size_t l = arch.size() - 1; l > 0; l--)
            {
                for (size_t j = 0; j < arch[l]; j++)
                {
                    // j = weight matrix row
                    // k = weight matrix col
                    T a = activations[l](0, j);
                    T da = g.activations[l](0, j);
                    T qa = a * (1 - a);
                    g.biases[l - 1](0, j) += 2 * da * qa;
                    for (size_t k = 0; k < arch[l - 1]; k++)
                    {
                        T pa = activations[l - 1](0, k);
                        T w = weights[l - 1](k, j);
                        g.weights[l - 1](k, j) += 2 * da * qa * pa;
                        g.activations[l - 1](0, k) += 2 * da * qa * w;
                    }
                }
            }
        }
        
        for (size_t i = 0; i < g.weights.size(); i++)
        {
            // sussy use of activate function :D
            // divide by the number of testcases

            g.weights[i] = g.weights[i].activate([n](auto x)
                                                 { return x / n; });
        }
        for (size_t i = 0; i < g.biases.size(); i++)
        {
            // sussy use of activate function :D
            // divide by the number of testcases

            g.biases[i] = g.biases[i].activate([n](auto x)
                                               { return x / n; });
        }

        // learning
        for (size_t i = 0; i < weights.size(); i++)
            for (size_t j = 0; j < weights[i].data.size(); j++)
                weights[i].data[j] -= rate * g.weights[i].data[j];

        for (size_t i = 0; i < biases.size(); i++)
            for (size_t j = 0; j < biases[i].data.size(); j++)
                biases[i].data[j] -= rate * g.biases[i].data[j];
    }
    void NeuralNetwork::finiteDiff(DataSet &train)
    {
        // the original cost
        T c = cost(train);

        std::vector<Matrix> W = weights;
        for (size_t i = 0; i < weights.size(); i++)
            for (size_t j = 0; j < weights[i].data.size(); j++)
            {
                // save the old value cause floating point imprecision
                T oldWeight = weights[i].data[j];

                weights[i].data[j] += eps;

                // (f(x + eps) - f(x)) / eps
                T dw = (cost(train) - c) / eps;

                // reset it back
                weights[i].data[j] = oldWeight;

                W[i].data[j] -= rate * dw;
            }

        std::vector<Matrix> B = biases;
        for (size_t i = 0; i < biases.size(); i++)
            for (size_t j = 0; j < biases[i].data.size(); j++)
            {
                // save the old value cause floating point imprecision
                T oldBias = biases[i].data[j];

                biases[i].data[j] += eps;

                // (f(x + eps) - f(x)) / eps
                T db = (cost(train) - c) / eps;

                // reset it back
                biases[i].data[j] = oldBias;

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
}