#include <vector>
#include "matrix.h"
class NN
{
private:
    typedef double T;
    std::vector<size_t> arch;
    std::vector<Matrix<T>> weights;

public:
    NN(std::vector<size_t> arch) : arch(arch)
    {
        weights.resize(arch.size() - 1);
        for (size_t i = 0; i < arch.size() - 1; i++)
            weights[i] = Matrix<T>{arch[i], arch[i + 1]};
    }

    void rand()
    {
        for (auto &w : weights)
            w.rand();
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
};