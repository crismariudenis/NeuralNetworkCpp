#include <iostream>
#include "../include/gym.h"
#include "../include/neuralnetwork.h"

typedef double T;
int main()
{
    //---- Setup network and data -------
    nn::NeuralNetwork n{{2,2, 1}};

    n.rand();
    // n.rate
    // n.
    nn::DataSet ds;
    ///-----------------------------------
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
    ///-----------------------------------
    nn::Gym gym(n);
    gym.train(ds, 100'000);

    for (size_t i = 0; i < ds.size(); i++)
    {
        nn::Matrix acc = n.forward(ds.getInputMat(i));
        std::vector<T> exp = ds.getData(i).output;
#define test ds.getInputMat(i).data
        std::cout << std::fixed << test[0] << " ^ " << test[1] << " = " << acc.data[0] << '\n';
    }
    n.printWeights();
    n.printBiases();
}