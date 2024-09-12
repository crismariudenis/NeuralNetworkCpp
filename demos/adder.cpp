#include <iostream>
#include "../include/gym.h"
#include "../include/neuralnetwork.h"

typedef double T;

size_t numbersLength = 3; // number of bits for each number
size_t maxNumber = 1 << numbersLength;
void generateData(nn::DataSet &ds)
{
    for (int i = 0; i < 20; i++)
    {
        int a = rand() % maxNumber;
        int b = rand() % maxNumber;
        std::vector<T> input;
        // add a and b 's bits

        for (int j = 0; j < numbersLength; j++)
            input.push_back((a >> j) & 1);

        for (int j = 0; j < numbersLength; j++)
            input.push_back((b >> j) & 1);

        std::vector<T> output;
        // add a + b 's bits
        for (int j = 0; j < numbersLength + 1; j++)
            output.push_back(((a + b) >> j) & 1);
        ds.addData(input, output);
    }
}
int main()
{
    //---- Setup network and data -------
    nn::NeuralNetwork n{{2 * numbersLength, 4 * numbersLength, numbersLength + 1}};

    nn::DataSet ds;
    generateData(ds);

    nn::Gym gym(n);
    gym.train(ds, 50'000);

    // ---------------------PRINT-------------------------
    T diff = 0;
    for (size_t i = 0; i < ds.size(); i++)
    {
        nn::Matrix acc = n.forward(ds.getInputMat(i));
        auto value = ds.getInputMat(i).data;
        size_t p = 1;

        double nr = 0;
        for (size_t i = 0, p = 1; i < numbersLength; i++, p *= 2)
            nr += value[i] * p;
        std::cout << nr << " + ";

        nr = 0;
        for (size_t i = 0, p = 1; i < numbersLength; i++, p *= 2)
            nr += value[i + numbersLength] * p;
        std::cout << nr << " = ";

        nr = 0;
        for (size_t i = 0, p = 1; i < numbersLength + 1; i++, p *= 2)
            nr += acc.data[i] * p;
        std::cout << nr << '\n';
    }
    // n.printWeights();
    // n.printBiases();
}