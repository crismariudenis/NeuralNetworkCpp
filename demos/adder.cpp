#include <iostream>
#include "../include/gym.h"
#include "../include/neuralnetwork.h"

typedef double T;

int main()
{
    //---- Setup network and data -------
    nn::NeuralNetwork n{{8, 16, 5}};

    n.rate = 1;
    n.nrSamples = 10;

    n.rand();
    nn::DataSet ds;
    
    ///-----------------------------------
    for (int i = 0; i < 60; i++)
    {
        int a = rand() % 8;
        int b = rand() % 8;
        std::vector<T> input;
        // add a and b 's bits

        for (int j = 0; j < 4; j++)
            input.push_back((a >> j) & 1);

        for (int j = 0; j < 4; j++)
            input.push_back((b >> j) & 1);

        std::vector<T> output;
        // add a + b 's bits
        for (int j = 0; j < 5; j++)
            output.push_back(((a + b) >> j) & 1);
        ds.addData(input, output);
    }
    ///-----------------------------------

    nn::Gym gym(n);
    gym.train(ds, 100000);

    // ---------------------PRINT-------------------------
    for (size_t i = 0; i < ds.size(); i++)
    {
        nn::Matrix acc = n.forward(ds.getInputMat(i));
        std::vector<T> exp = ds.getData(i).output;
#define test ds.getInputMat(i).data
        std::cout << std::fixed << test[0] + 2 * test[1] + 4 * test[2] + 8 * test[3] + 16 * test[4] << " + "
                  << test[4] + 2 * test[5] + 4 * test[6] + 8 * test[7] + 16 * test[8] << " = " << acc.data[0] + 2 * acc.data[1] + 4 * acc.data[2] + 8 * acc.data[3] + 16 * acc.data[4] + 32 * acc.data[5] << '\n';
    }
    n.printWeights();
    n.printBiases();
}