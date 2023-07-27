#include <iostream>
#include "../src/gym.h"
#include "../src/neuralnetwork.h"

typedef double T;
int main()
{
    //---- Setup network and data -------
    nn::NeuralNetwork n{{6, 12, 4}};
    n.rand();

    nn::DataSet<T> train;
    train.generateData();
    ///-----------------------------------

    Gym<T> gym(n);
    gym.train(train);

    for (size_t i = 0; i < train.size(); i++)
    {
        nn::Matrix<T> acc = n.forward(train.getInputMat(i));
        std::vector<T> exp = train.getData(i).output;
#define test train.getInputMat(i).data
        std::cout << std::fixed << test[0] + 2 * test[1] + 4 * test[2] << " + " << test[3] + 2 * test[4] + 4 * test[5] << " = " <<  acc.data[0] + 2 * acc.data[1] + 4 * acc.data[2] + 8*acc.data[3] << '\n';
    }
    n.printWeights();
    n.printBiases();
}