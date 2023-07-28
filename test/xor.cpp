#include <iostream>
#include <vector>
#include "../src/neuralnetwork.h"
#include "../src/dataset.h"
#include "../src/gym.h"

typedef double T;

int main()
{
    nn::NeuralNetwork n({2, 2, 1});
    n.rand();
    nn::DataSet<T> train{"test.txt"};
    train.generateData();

    nn::Gym<T> gym(n, 100'000);
    gym.train(train);

    for (size_t i = 0; i < train.size(); i++)
    {
        nn::Matrix<T> acc = n.forward(train.getInputMat(i));
        std::vector<T> exp = train.getData(i).output;
        std::cout << std::fixed << train.getInputMat(i).data[0] << " ^ " << train.getInputMat(i).data[1] << " = " << acc.data[0] << '\n';
    }
    n.printWeights();
    n.printBiases();
}