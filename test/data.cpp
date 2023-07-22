#include <iostream>
#include "../src/data.h"

int main()
{
    typedef double T;
    Data<T> train{"test.txt"};
    std::vector<std::vector<T>> t = {
        {0, 0},
        {1, 2},
        {2, 4},
        {3, 6},
        {4, 8},
    };
    for (auto &x : t)
        train.addData({x[0]}, {x[1]});

    // loop through data
    for (size_t i = 0; i < train.size(); i++)
    {
        DataPoint<T> &dataPoint = train.getData(i);
    }

    train.print();
}