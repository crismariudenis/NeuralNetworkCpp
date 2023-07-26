#include <iostream>
#include <vector>
#include "../src/neuralnetwork.h"
#include "../src/dataset.h"

typedef double T;
nn::DataSet<T> train{"test.txt"};

int main()
{
    nn::NeuralNetwork n({2, 2, 1});

    train.generateData();
    n.rand();

    for (size_t epoch = 1; epoch <= 100'000; epoch++)
    {
        n.finiteDiff(train);
        if (epoch % 10 == 0)
            std::cout << "cost: " << n.cost(train) << '\n';
    }

    for (size_t i = 0; i < train.size(); i++)
    {
        nn::Matrix<T> acc = n.forward(train.getInputMat(i));
        std::vector<T> exp = train.getData(i).output;
        std::cout << std::fixed << train.getInputMat(i).data[0] << " ^ " << train.getInputMat(i).data[1] << " = " << acc.data[0] << '\n';
    }
    n.printWeights();
    n.printBiases();
}