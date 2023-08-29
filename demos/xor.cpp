#include <iostream>
#include "../include/gym.h"
#include "../include/neuralnetwork.h"

typedef double T;
void generateData(nn::DataSet &ds)
{
    const std::vector<std::vector<T>> t = {
        {1, 1, 0},
        {1, 0, 1},
        {0, 1, 1},
        {0, 0, 0},
    };

    for (const auto &x : t)
    {
        std::vector<T> input{x[0], x[1]};
        std::vector<T> output{x[2]};
        ds.addData(input, output);
    }
}
int main()
{
    //---- Setup network and data -------
    nn::NeuralNetwork n{{2, 2, 1}};
    n.rand();

    nn::DataSet ds;
    generateData(ds);

    nn::Gym gym(n);
    gym.train(ds, 50'000);
    
    // ---------------------PRINT-------------------------
    for (size_t i = 0; i < ds.size(); i++)
    {
        nn::Matrix acc = n.forward(ds.getInputMat(i));
        auto value = ds.getInputMat(i).data;
        std::cout << std::fixed << value[0] << " ^ " << value[1] << " = " << acc.data[0] << '\n';
    }
    // n.printWeights();
    // n.printBiases();
}