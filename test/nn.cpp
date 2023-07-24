#include <iostream>
#include <vector>
#include "../src/nn.h"
#include "../src/dataset.h"

typedef double T;
DataSet<T> train{"test.txt"};

int main()
{
    NN nn({2, 1});

    train.generateData();
    nn.rand();
    nn.printWeights();
    nn.printBiases();

    // train.getInput(1).printX("input0");

    std::cout << "matrix cost: " << nn.cost(train) << '\n';
}