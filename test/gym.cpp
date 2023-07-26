#include <iostream>
#include "../src/gym.h"
#include "../src/neuralnetwork.h"

typedef double T;
int main()
{
    //---- Setup network and data -------
    nn::NeuralNetwork n{{2, 2, 1}};
    n.rand();

    nn::DataSet<T> train;
    train.generateData();
    ///-----------------------------------

    Gym<T> gym(n);
    gym.train(train);

   
        //! Pass n by reference
    for (size_t i = 0; i < train.size(); i++)
    {
        nn::Matrix<T> acc = n.forward(train.getInputMat(i));
        std::vector<T> exp = train.getData(i).output;
        std::cout << std::fixed << train.getInputMat(i).data[0] << " ^ " << train.getInputMat(i).data[1] << " = " << acc.data[0] << '\n';
    }
    n.printWeights();
    n.printBiases();
}