#include <iostream>
#include "raylib.h"
#include "../include/gym.h"
#include "../include/neuralnetwork.h"

Image image;
void generateData(nn::DataSet &ds)
{
    image = LoadImage("./data/9.png"); // Load image data into CPU memory (RAM)

    Color *colors = LoadImageColors(image);
    for (int y = 0; y < image.height; y++)
        for (int x = 0; x < image.width; x++)
        {
            int index = (y * image.width) + x;
            int nr = 3;
            nn::T val = colors[index].r;
            if (val != 0)
                nr = 3 - (int)std::log10(val);

            std::vector<nn::T> input(2);
            input[0] = float(x) / (image.width - 1.0);
            input[1] = float(y) / (image.height - 1.0);

            std::vector<nn::T> output(1);
            output[0] = (nn::T)val / 255.0;
            ds.addData(input, output);
        }
}
int main()
{
    nn::NeuralNetwork n{{2, 7, 7, 1}};
    n.rand();
    n.nrSamples = 28;

    nn::Gym gym(n, nn::Gym::Mode::UPSCALE);

    nn::DataSet ds;
    generateData(ds);
    ds.shuffle();

    gym.train(ds, 20'000);
}