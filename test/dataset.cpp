#include <iostream>
#include "../src/dataset.h"

int main()
{
    typedef double T;
    DataSet<T> train{"test.txt"};
    const std::vector<std::vector<T>> t = {
        {0, 0},
        {1, 2},
        {2, 4},
        {3, 6},
        {4, 8},
    };

    for (const auto &x : t)
    {
        std::vector<T> input{x[0]};
        std::vector<T> output{x[1]};
        train.addData(input, output);
    }

    train.getOutputMat(3).printX();

    // train.print();
}