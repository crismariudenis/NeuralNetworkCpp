
#pragma once
#include <thread>
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
        std::vector<Matrix> v_weights; // Momentum weights
        std::vector<Matrix> v_biases;  // Momentum biases

        double initialRate = 1;
        double decayRate = 0;
        size_t nrSamples = 1;

    public:
        double momentum = 0;
        double rate;
        bool isRandomizing = false;

        double eps = 1e-3;
        T lastCost;

        NeuralNetwork(std::vector<size_t> arch);
        void rand();
        Matrix forward(Matrix input);
        T cost(DataSet &ds);

        void train(DataSet &ds, size_t epoch);
        template <typename Iterator>
        void backProp(Iterator start, Iterator end);
        void finiteDiff(DataSet &ds);

        void printArch();
        void printWeights();
        void printBiases();

        std::vector<size_t> getArch();
        T getBias(size_t layer, size_t nr);
        T getWeight(size_t layer1, size_t node1, size_t node2);
        void copyWeightsAndBiasesFrom(const NeuralNetwork &other);
        void setHyperparameters(double initialRate, double decayRate, double momentum, size_t nrSamples);
    };
    NeuralNetwork::NeuralNetwork(std::vector<size_t> arch) : arch(arch)
    {
        weights.resize(arch.size() - 1);
        biases.resize(arch.size() - 1);
        activations.resize(arch.size());

        v_weights.resize(arch.size() - 1);
        v_biases.resize(arch.size() - 1);

        for (size_t i = 0; i < arch.size() - 1; i++)
        {
            weights[i] = Matrix{arch[i], arch[i + 1]};
            v_weights[i] = Matrix{arch[i], arch[i + 1]};
        }

        for (size_t i = 0; i < arch.size() - 1; i++)
        {
            biases[i] = Matrix{1, arch[i + 1]};
            v_biases[i] = Matrix{1, arch[i + 1]};
        }

        for (size_t i = 0; i < arch.size(); i++)
            activations[i] = Matrix{1, arch[i]};

        rand(); // Randomizing the weights and biases
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
            (input * weights[i] + biases[i]).activate([](auto x)
                                                      { return 1.0 / (1.0 + exp(-x)); });

            // Sussy behaviour if I replace input with activations
        }
        activations.back() = input;

        return input;
    }
    T NeuralNetwork::cost(DataSet &ds)
    {
        T cost = 0;
        for (size_t i = 0; i < ds.size(); i++)
        {
            // actual value
            Matrix act = forward(ds.getInputMat(i));

            // expected value
            Matrix exp = ds.getOutputMat(i);

            for (size_t j = 0; j < act.data.size(); j++)
            {
                T d = act(0, j) - exp(0, j);
                cost += d * d;
            }
        }
        lastCost = cost / static_cast<T>(ds.size());
        return cost / static_cast<T>(ds.size());
    }
    void NeuralNetwork::train(DataSet &ds, size_t epoch)
    {
        assert(nrSamples <= ds.size());
        auto start = ds.begin();
        auto end = ds.end();
        size_t remaining = ds.size();
        // isRandomizing = true;
        // ds.shuffle();
        // isRandomizing = false;

        // Stochastic Gradient Descent
        // dividing the dataset in equal batches
        rate = initialRate / (1 + decayRate * epoch);

        for (size_t i = 0; i < nrSamples; i++)
        {
            size_t sampleSize = remaining / (nrSamples - i);
            backProp(start, std::min(end, start + sampleSize));
            start += sampleSize;
            remaining -= sampleSize;
        }
    }
    template <typename Iterator>
    void NeuralNetwork::backProp(Iterator start, Iterator end)
    {
        assert(start < end);
        size_t n = end - start;

        // Declare and initialize gradient variables
        std::vector<Matrix> g_weights(weights.size());
        std::vector<Matrix> g_biases(biases.size());
        std::vector<Matrix> g_activations(arch.size());

        for (size_t i = 0; i < weights.size(); ++i)
        {
            auto &[w_rows, w_cols] = weights[i].shape;
            auto &[b_rows, b_cols] = biases[i].shape;

            g_weights[i] = Matrix{w_rows, w_cols};
            g_biases[i] = Matrix{b_rows, b_cols};
        }

        for (size_t i = 0; i < arch.size(); ++i)
            g_activations[i] = Matrix{1, arch[i]};

        // l = current layer
        // i = current activation
        // j = previous activation
        for (Iterator it = start; it != end; ++it)
        {
            DataPoint &dp = *it;

            // expected value
            Matrix exp = dp.getOutputMat();

            forward(dp.getInputMat());

            // reset the activations
            for (auto &a : g_activations)
                a.fill(0);

            for (size_t i = 0; i < exp.data.size(); i++)
                g_activations.back()(0, i) = activations.back()(0, i) - exp(0, i);

            for (size_t l = arch.size() - 1; l > 0; l--)
            {
                for (size_t i = 0; i < arch[l]; i++)
                {
                    // i = weight matrix row
                    // j = weight matrix col
                    T a = activations[l](0, i);
                    T da = g_activations[l](0, i);
                    T qa = a * (1 - a);
                    g_biases[l - 1](0, i) += 2 * da * qa;
                    for (size_t j = 0; j < arch[l - 1]; j++)
                    {
                        T pa = activations[l - 1](0, j);
                        T w = weights[l - 1](j, i);
                        g_weights[l - 1](j, i) += 2 * da * qa * pa;
                        g_activations[l - 1](0, j) += 2 * da * qa * w;
                    }
                }
            }
        }

        // Update weights and biases with momentum
        for (size_t l = 0; l < weights.size(); ++l)
        {
            // Average the gradients inline
            g_weights[l].activate([n](auto x)
                                  { return x / n; });
            g_biases[l].activate([n](auto x)
                                 { return x / n; });

            v_weights[l] = v_weights[l] * momentum - g_weights[l] * rate;
            v_biases[l] = v_biases[l] * momentum - g_biases[l] * rate;
            weights[l] += v_weights[l];
            biases[l] += v_biases[l];
        }
    }
    void NeuralNetwork::finiteDiff(DataSet &ds)
    {
        // the original cost
        T c = cost(ds);

        std::vector<Matrix> W = weights;
        for (size_t i = 0; i < weights.size(); i++)
            for (size_t j = 0; j < weights[i].data.size(); j++)
            {
                // save the old value cause floating point imprecision
                T oldWeight = weights[i].data[j];

                weights[i].data[j] += eps;

                // (f(x + eps) - f(x)) / eps
                T dw = (cost(ds) - c) / eps;

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
                T db = (cost(ds) - c) / eps;

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

    std::vector<size_t> NeuralNetwork::getArch() { return arch; }
    T NeuralNetwork::getBias(size_t layer, size_t nr)
    {
        return biases[layer](0, nr);
    }
    T NeuralNetwork::getWeight(size_t layer1, size_t node1, size_t node2)
    {
        return weights[layer1](node1, node2);
    }

    void NeuralNetwork::copyWeightsAndBiasesFrom(const NeuralNetwork &other)
    {
        assert(this->arch == other.arch);
        this->weights = other.weights;
        this->biases = other.biases;
    }

    void NeuralNetwork::setHyperparameters(double initialRate, double decayRate, double momentum, size_t nrSamples)
    {
        this->initialRate = initialRate;
        this->decayRate = decayRate;
        this->momentum = momentum;
        this->nrSamples = nrSamples;
    }

}