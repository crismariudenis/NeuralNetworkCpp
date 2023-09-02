#include <iostream>
#include "raylib.h"
#include "../include/gym.h"

Image image1, image2;
void generateData(nn::DataSet &ds)
{

    // Image 1
    image1 = LoadImage("./data/9.png");

    Color *colors = LoadImageColors(image1);
    for (int y = 0; y < image1.height; y++)
        for (int x = 0; x < image1.width; x++)
        {
            int index = (y * image1.width) + x;
            int nr = 3;
            nn::T val = colors[index].r;
            if (val != 0)
                nr = 3 - (int)std::log10(val);

            std::vector<nn::T> input(3);
            input[0] = float(x) / (image1.width - 1.0);
            input[1] = float(y) / (image1.height - 1.0);
            input[2] = 0.0;

            std::vector<nn::T> output(1);
            output[0] = (nn::T)val / 255.0;
            ds.addData(input, output);
        }

    // Image 2
    image2 = LoadImage("./data/8.png");

    Color *colors2 = LoadImageColors(image2);
    for (int y = 0; y < image2.height; y++)
        for (int x = 0; x < image2.width; x++)
        {
            int index = (y * image2.width) + x;
            int nr = 3;
            nn::T val = colors2[index].r;
            if (val != 0)
                nr = 3 - (int)std::log10(val);

            std::vector<nn::T> input(3);
            input[0] = float(x) / (image2.width - 1.0);
            input[1] = float(y) / (image2.height - 1.0);
            input[2] = 1.0;

            std::vector<nn::T> output(1);
            output[0] = (nn::T)val / 255.0;
            ds.addData(input, output);
        }
    UnloadImage(image1), UnloadImage(image2);
}
int main()
{
    nn::NeuralNetwork n{{3, 9, 9, 1}};
    n.rand();
    n.nrSamples = 4 * 28;

    nn::Gym gym(n, nn::Gym::Mode::TRANSITION);

    nn::DataSet ds;
    generateData(ds);

    ds.shuffle();

    gym.train(ds, 10'000);
}