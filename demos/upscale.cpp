#include <iostream>
#include "raylib.h"
#include "../include/gym.h"
#include "../include/neuralnetwork.h"

Image image;
#define UPSCALE 
void generateData(nn::DataSet &ds)
{
    image = LoadImage("./data/8.png"); // Load image data into CPU memory (RAM)

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
void display(nn::NeuralNetwork &n)
{
    size_t outWidth = 512;
    size_t outHeight = 512;
    const char *outPath = "upscaled.png";
    uint8_t *outPixels = (uint8_t *)malloc(sizeof(*outPixels) * outHeight * outWidth);
    for (int y = 0; y < outHeight; y++)
        for (int x = 0; x < outWidth; x++)
        {
            nn::Matrix input{1, 2};
            input(0, 0) = nn::T(x) / (outWidth - 1.0);
            input(0, 1) = nn::T(y) / (outHeight - 1.0);
            nn::Matrix output = n.forward(input);

            uint8_t pixel = (uint8_t)(output(0, 0) * 255.0);
            outPixels[y * outWidth + x] = pixel;
        }
    if (!stbi_write_png(outPath, outWidth, outHeight, 1, outPixels, outWidth * (sizeof(*outPixels))))
    {
        std::cerr << "ERROR: could not save image\n"
                  << outPath;
        exit(0);
    }
    free(outPixels);
}
int main()
{
    nn::NeuralNetwork n{{2, 28, 1}};
    n.rand();
    n.nrSamples = 4 * 28;

    nn::DataSet ds;
    generateData(ds);
    ds.shuffle();

    nn::Gym gym(n);
    gym.train(ds, 50'000);

    display(n);
}