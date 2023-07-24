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
    for (size_t epoch = 1; epoch <= 100'000; epoch++)
    {
        nn.finiteDiff(train);
        if (epoch % 10 == 0)
            std::cout << "cost: " << nn.cost(train) << '\n';
    }

    for (size_t i = 0; i < train.size(); i++)
    {
        Matrix<T> acc = nn.forward(train.getInput(i));
        std::vector<T> exp = train.getData(i).output;
        std::cout<<std::fixed<< train.getInput(i).data[0] << " & " << train.getInput(i).data[1] << " = " << acc.data[0] << '\n';
    }
    nn.printWeights();
    nn.printBiases();
}