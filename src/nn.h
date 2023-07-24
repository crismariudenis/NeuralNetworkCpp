#include <vector>
#include "dataset.h"
#include "matrix.h"
class NN
{
private:
    typedef double T;
    std::vector<size_t> arch;
    std::vector<Matrix<T>> biases;
    std::vector<Matrix<T>> weights;
    const double eps = 1e-3;
    const double rate = 1e-1;

public:
    NN(std::vector<size_t> arch) : arch(arch)
    {
        weights.resize(arch.size() - 1);
        biases.resize(arch.size() - 1);
        for (size_t i = 0; i < arch.size() - 1; i++)
            weights[i] = Matrix<T>{arch[i], arch[i + 1]};

        for (size_t i = 0; i < arch.size() - 1; i++)
            biases[i] = Matrix<T>{1, arch[i + 1]};
    }

    void rand()
    {
        for (auto &w : weights)
            w.rand();

        for (auto &b : biases)
            b.rand();
    }
    Matrix<T> forward(Matrix<T> input)
    {

        for (size_t i = 0; i < weights.size(); i++)
        {
            // forward the input add activate using sigmoid function
            input = (input * weights[i] + biases[i]).activate([](auto x){ return 1.0 / (1.0 + exp(-x)); });
        }
        // std::cout << " forwards.out: ";
        // for(auto x:input.data){
        //     std::cout << x << " ";
        // }
        // std::cout << '\n';
        return input;
    }
    Matrix<T> forward(std::vector<T> v)
    {
        Matrix<T> m{1, v.size()};
        m.data = v;
        return forward(m);
    }
    T cost(DataSet<T> &train)
    {
        T cost = 0;
        double eps = 1e-3;
        double rate = 1e-3;

        for (size_t i = 0; i < train.size(); i++)
        {
            // actual value
            Matrix<T> act = forward(train.getData(i).input);

            // expected value
            Matrix<T> exp{1, train.getData(i).output.size()};
            exp.data = train.getData(i).output;

            assert(act.shape == exp.shape);
            for (size_t j = 0; j < act.data.size(); j++)
            {
                T d = act.data[j] - exp.data[j];
                cost += d * d;
            }
        }
        return cost / static_cast<T>(train.size());
    }
    void finiteDiff(DataSet<T> &train)
    {
        auto c = cost(train);

        //
        std::vector<Matrix<T>> W = weights;
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

        std::vector<Matrix<T>> B = biases;
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

    void printArch()
    {
        std::cout << "NN Arch([";
        for (size_t i = 0; i < arch.size() - 1; i++)
            std::cout << arch[i] << ',';
        std::cout << arch.back() << "])\n";
    }

    void printWeights()
    {
        std::cout << "NN = [\n";
        for (size_t i = 0; i < weights.size(); i++)
        {
            std::cout << "  ";
            weights[i].printX("w" + std::to_string(i));
        }
        std::cout << "]\n";
    }
    void printBiases()
    {
        std::cout << "NN = [\n";
        for (size_t i = 0; i < biases.size(); i++)
        {
            std::cout << "  ";
            biases[i].printX("b" + std::to_string(i));
        }
        std::cout << "]\n";
    }
};