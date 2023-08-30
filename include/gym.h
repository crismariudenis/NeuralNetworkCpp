#pragma once
#include <iostream>
#include "neuralnetwork.h"
#include "raylib.h"
#include <thread>
#include <condition_variable>
#include <mutex>

#include "stb_image_write.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

namespace nn
{
    class Gym
    {
    private:
        class ImageTexture
        {
        private:
            size_t x, y, w, h;
            int screenWidth, screenHeight;
            const char *path;
            Image img;
            Texture2D tex;

        public:
            ImageTexture(size_t screenWidth, size_t screenHeight, size_t x, size_t y, size_t w, size_t h, const char *path) : screenWidth(screenWidth), screenHeight(screenHeight), x(x), y(y), w(w), h(h), path(path)
            {
            }
            ImageTexture()
            {
            }
            void load()
            {
                img = LoadImage(path); // Load image data into CPU memory (RAM)
                ImageResizeNN(&img, w, h);
                tex = LoadTextureFromImage(img); // Image converted to texture, GPU memory (RAM -> VRAM)
                UnloadImage(img);                // Unload image data from CPU memory (RAM)
                img = LoadImageFromTexture(tex); // Load image from GPU texture (VRAM -> RAM)
                UnloadTexture(tex);              // Unload texture from GPU memory (VRAM)
                tex = LoadTextureFromImage(img); // Recreate texture from retrieved image data (RAM -> VRAM)
                UnloadImage(img);
            }
            void draw()
            {
                DrawTexture(tex, x, y, WHITE);
            }
            void unload()
            {
                UnloadTexture(tex);
            }
        };
        // Graphic data
        Color backgroundColor{0x18, 0x18, 0x18};
        const int scale = 80;
        const int screenWidth = 16 * scale;
        const int screenHeight = 9 * scale;
        const char *windowName = "Neural Network";

        // For the digit recognition
#ifdef UPSCALE
        Image inputImg, outputImg;
        Texture2D inputTex, outputTex;
        const int imageWidth = 200;
        const int imageHeight = 200;
        const int nrWindows = 3;
        const char *outputPath = "./data/up9.png";
        const char *inputPath = "./data/9.png";
        ImageTexture out, in;
#else
        const int nrWindows = 2;
#endif

        //  For the NeuralNetwork
        nn::NeuralNetwork &n;
        size_t epochs = 100;
        size_t epoch = 0;
        std::vector<T> costs;
        nn::DataSet ds;

        // For threads communication
        bool paused = false;
        bool restarted = false;
        bool closed = false;

        // For paussing the threads while the network randomises
        std::mutex m;
        std::condition_variable cv;

    public:
        Gym(nn::NeuralNetwork &n);

        void setup();
        void train(nn::DataSet ds, size_t epochs);
        void computing();
        void drawing();
        void plotCost();
        void drawNetwork();
        void upscale();
        void drawImages();
    };
    Gym::Gym(nn::NeuralNetwork &n) : n(n)
    {
        setup();
    }
    void Gym::train(nn::DataSet ds, size_t epochs)
    {
        this->epochs = epochs;
        this->ds = ds;

        costs.resize(epochs);

        // thread computing cause drawing is slow
        std::thread t1(&Gym::computing, this);
        std::thread t2(&Gym::upscale, this);
        drawing();
        t2.join();
        t1.join();
    }

    void Gym::computing()
    {

        for (epoch = 0; epoch < epochs; epoch++)
        {
            if (restarted)
            {

                // locks the main threads until
                std::unique_lock<std::mutex> lk(m);
                // network resets
                n.rand();
                epoch = 0;
                lk.unlock();
                cv.notify_one();
                restarted = false;
            }
            if (closed)
                return;
            while (paused && IsKeyUp(KEY_SPACE))
            {
            }

            costs[epoch] = n.cost(ds);

            n.train(ds);
        }
    }
    void Gym::setup()
    {
        SetTraceLogLevel(LOG_NONE);
        InitWindow(screenWidth, screenHeight, windowName);
        SetTargetFPS(60);
#ifdef UPSCALE
        size_t pad = 5;
        in = ImageTexture{screenWidth, screenHeight, screenWidth - 400 -pad, 2 * screenHeight / 3 - 200, 200, 200, inputPath};
        in.load();

        out = ImageTexture{screenWidth, screenHeight, screenWidth - 200 - pad, 2 * screenHeight / 3 - 200, 200, 200, outputPath};
        out.load();
#endif
    }
    void Gym::upscale()
    {
#ifndef UPSCALE
        return;
#endif
        int nr = 0;
        while (true and !closed)
        {
            if (nr % 10 == 0)
            {
                NeuralNetwork m = n;
                uint8_t *outPixels = (uint8_t *)malloc(sizeof(*outPixels) * imageHeight * imageWidth);
                for (int y = 0; y < imageHeight; y++)
                    for (int x = 0; x < imageWidth; x++)
                    {
                        nn::Matrix input{1, 2};
                        input(0, 0) = nn::T(x) / (imageWidth - 1.0);
                        input(0, 1) = nn::T(y) / (imageHeight - 1.0);
                        nn::Matrix output = m.forward(input);

                        uint8_t pixel = (uint8_t)(output(0, 0) * 255.0);
                        outPixels[y * imageHeight + x] = pixel;
                    }
                stbi_write_png(outputPath, imageWidth, imageHeight, 1, outPixels, imageWidth * (sizeof(*outPixels)));
                free(outPixels);
            }
            nr++;
        }
    }
    void Gym::drawImages()
    {
    }
    void Gym::drawing()
    {
        while (!WindowShouldClose())
        {

            BeginDrawing();
            {
                ClearBackground(backgroundColor);
                if (IsKeyPressed(KEY_SPACE))
                    paused = !paused;
                if (IsKeyPressed(KEY_R))
                {
                    restarted = true;
                    {
                        std::lock_guard<std::mutex> lk(m);
                    }
                    cv.notify_one();
                    {
                        std::unique_lock<std::mutex> lk(m);
                    }
                }

                plotCost();
                char buffer[64];
                snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Cost: %f\n", epoch, epochs, n.rate, n.cost(ds));
                DrawText(buffer, 5, 0, 30, WHITE);
                drawNetwork();
                out.load();
                out.draw();
                in.draw();
            }
            EndDrawing();
        }
#ifdef UPSCALE
        UnloadTexture(inputTex);
        UnloadTexture(outputTex);
#endif
        closed = true;
        CloseWindow();
    }
    void Gym::drawNetwork()
    {
        float h = screenHeight / 1.5;
        float w = screenWidth / nrWindows;
        float rectX = screenWidth / nrWindows;
        float rectY = (screenHeight - h) / 2;
        float padX = w / 20;
        float padY = 10;
        rectX += padX;
        w -= 2 * padX;

        // compute pozitions of nodes
        auto arch = n.getArch();
        std::vector<Vector2> nodes[arch.size()];
        float offX = w / (arch.size() - 1);
        for (size_t i = 0; i < arch.size(); i++)
        {
            float offY = h / (arch[i] + 1);
            nodes[i].resize(arch[i]);
            for (size_t j = 0; j < arch[i]; j++)
                nodes[i][j] = {rectX + i * offX, rectY + (j + 1) * offY};
        }

        Color highColor = DARKBLUE;
        Color lowColor = RED;
        // draw edges
        for (size_t l = 0; l < arch.size() - 1; l++)
            for (size_t a = 0; a < arch[l]; a++)
                for (size_t b = 0; b < arch[l + 1]; b++)
                {
                    T val = n.getWeight(l, a, b);
                    auto sigVal = 1 / (1 + exp(-val));
                    highColor.a = floor(255.0 * sigVal);
                    DrawLineEx(nodes[l][a], nodes[l + 1][b], 2, ColorAlphaBlend(lowColor, highColor, WHITE));
                }
        // draw nodes
        for (size_t i = 0; i < arch.size(); i++)
            for (size_t j = 0; j < arch[i]; j++)
            {
                if (i == 0)
                    DrawCircle(nodes[i][j].x, nodes[i][j].y, 10, GRAY);
                else
                {
                    T val = n.getBias(i - 1, j);
                    auto sigVal = 1 / (1 + exp(-val));
                    highColor.a = floor(255.f * sigVal);
                    DrawCircle(nodes[i][j].x, nodes[i][j].y, 10, ColorAlphaBlend(lowColor, highColor, WHITE));
                }
            }
    }
    void Gym::plotCost()
    {

        if (epoch == 0)
            return;

        T rectX = 0;
        T h = screenHeight / 1.5;
        T w = screenWidth / nrWindows;
        T rectY = (screenHeight - h) / 2;

        T offX = w / epoch;
        T offY = h / costs[1];
        T lastX = 0;
        T lastY = 0;

        // Base line
        DrawRectangle(0, rectY + h, w, 2, WHITE);
        DrawText("0", 0, rectY + h - 20, 20, WHITE);

        for (size_t i = 0; i < epoch; i++)
        {
            if (i != 0)
            {
                DrawLine(rectX + lastX, rectY + lastY, rectX + lastX + offX, rectY + (costs[1] - costs[i]) * offY, RED);
                lastX = lastX + offX;
                lastY = (costs[1] - costs[i]) * offY;
            }
        }
    }
}
