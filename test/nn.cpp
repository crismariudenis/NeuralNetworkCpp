#include <iostream>
#include <vector>
#include "../src/nn.h"

std::vector<std::vector<float>> train = {
    {0, 0},
    {1, 2},
    {3, 6},
    {4, 8},
};
int main()
{
    NN nn({10,3,5});
    nn.rand();
    nn.printWeights();
}