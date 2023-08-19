#include <iostream>
#include "../include/gym.h"
#include "../include/neuralnetwork.h"

typedef double T;
int main()
{
    //---- Setup network and data -------
    nn::NeuralNetwork n{{2,2, 1}};

    n.rate = 1;
    n.eps = 1e-3;

    n.rand();
    // n.rate
    // n.
    nn::DataSet train;
    train.generateData();
    ///-----------------------------------

    nn::Gym gym(n);
    gym.train(train, 100'000);

    for (size_t i = 0; i < train.size(); i++)
    {
        nn::Matrix acc = n.forward(train.getInputMat(i));
        std::vector<T> exp = train.getData(i).output;
#define test train.getInputMat(i).data
        std::cout << std::fixed << test[0] << " ^ " << test[1] << " = " << acc.data[0] << '\n';
    }
    n.printWeights();
    n.printBiases();
}