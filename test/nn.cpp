#include <iostream>
#include <vector>

#include "../src/nn.h"
#include "../src/dataset.h"

typedef double T;
int main()
{
    DataSet<T> train{"test.txt"};
    train.generateData();

    NN nn({10, 3, 5});
    nn.rand();
    nn.printWeights();
    nn.printBiases();

    // float w = random(1, 2);
    // for (const auto &x : train.data)
    // {
    //     T act = x.input[0] * w;
    //     T exp = x.output[0];
    //     std::cout<< "actual: " << act << ", expected: " <<exp << '\n';
    // }
}