#pragma once
#include <iostream>
#include "neuralnetwork.h"
#include "../include/raylib.h"
#include <thread>
template <typename T>
class Gym
{
private:
    nn::NeuralNetwork &n;

    Color backgroundColor{0x18, 0x18, 0x18};
    const int scale = 80;
    const int screenWidth = 16 * scale;
    const int screenHeight = 9 * scale;
    const char *windowName = "Neural Network";
    size_t maxEpoch = 100'000;
    size_t epoch = 0;
    std::vector<T> costs;
    nn::DataSet<T> data;
    bool paused = false;

public:
    Gym(nn::NeuralNetwork &n) : n(n)
    {
        setup();
        costs.resize(maxEpoch);
    }

    void setup()
    {
        SetTraceLogLevel(LOG_ERROR);
        InitWindow(screenWidth, screenHeight, windowName);
        SetTargetFPS(60);
    }

    void train(nn::DataSet<T> train)
    {
        data = train;

        // thread computing cause drawing is slow
        std::thread t(&Gym<T>::computing, this);
        drawing();
        t.join();
    }

    void computing()
    {
        for (epoch = 0; epoch < maxEpoch; epoch++)
        {
            while (paused)
            {
            }

            costs[epoch] = n.cost(data);
            n.finiteDiff(data);
        }
    }

    void drawing()
    {
        while (!WindowShouldClose())
        {

            if (IsKeyPressed(KEY_SPACE))
                paused = !paused;

            BeginDrawing();
            {
                ClearBackground(backgroundColor);
                plotCost();
                char buffer[64];
                snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Cost: %f\n", epoch, maxEpoch, n.cost(data));
                DrawText(buffer, 5, 0, 30, WHITE);
            }
            EndDrawing();
        }
        CloseWindow();
    }
    void plotCost()
    {
        if (epoch == 0)
            return;

        double rectX = 0; // should be 0
        double rectY = screenHeight / 4.0;
        double h = screenHeight / 2;
        double w = screenWidth / 2;
        // DrawRectangle(rectX, rectY, w, h, Color{245, 0, 0, 10});

        double offX = w / epoch;
        double offY = h / costs[0];
        float lastX = 0;
        float lastY = 0;

        // Base line
        DrawRectangle(0, rectY + h, w, 2, WHITE);
        DrawText("0", 0, rectY + h - 20, 20, WHITE);

        for (size_t i = 0; i < epoch; i++)
        {
            if (i != 0)
            {
                DrawLine(rectX + lastX, rectY + lastY, rectX + lastX + offX, rectY + (costs[0] - costs[i]) * offY, RED);
                lastX = lastX + offX;
                lastY = (costs[0] - costs[i]) * offY;
            }
        }
    }
};
