#include <iostream>
#include "../include/gym.h"
#include "../include/neuralnetwork.h"

typedef double T;
int main()
{
    //---- Setup network and data -------
    nn::NeuralNetwork n{{8, 16, 5}, 1, 1e-3};
    n.rand();
    // n.rate
    // n.
    nn::DataSet train;
    train.generateData();
    ///-----------------------------------

    nn::Gym gym(n);
    gym.train(train, 10'000);

    for (size_t i = 0; i < train.size(); i++)
    {
        nn::Matrix acc = n.forward(train.getInputMat(i));
        std::vector<T> exp = train.getData(i).output;
#define test train.getInputMat(i).data
        std::cout << std::fixed << test[0] + 2 * test[1] + 4 * test[2] + 8 * test[3] << " + "
                  << test[4] + 2 * test[5] + 4 * test[6] + 8 * test[7] << " = " << acc.data[0] + 2 * acc.data[1] + 4 * acc.data[2] + 8 * acc.data[3] + 16 * acc.data[4] << '\n';
    }
    n.printWeights();
    n.printBiases();
}