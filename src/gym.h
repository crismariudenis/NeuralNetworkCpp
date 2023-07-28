#pragma once
#include <iostream>
#include "neuralnetwork.h"
#include "../include/raylib.h"
#include <thread>
#include <condition_variable>
#include <mutex>
namespace nn
{
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
        size_t maxEpoch = 100;
        size_t epoch = 0;
        std::vector<T> costs;
        nn::DataSet<T> data;
        bool paused = false;
        bool restarted = false;

        // For paussing the threads while the network randomises
        std::mutex m;
        std::condition_variable cv;

    public:
        Gym(nn::NeuralNetwork &n, size_t maxEpoch = 0) : n(n), maxEpoch(maxEpoch)
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
                if (restarted)
                {

                    // locks the main threads until
                    std::unique_lock lk(m);
                    // network resets
                    n.rand();
                    epoch = 0;
                    lk.unlock();
                    cv.notify_one();
                    restarted = false;
                }
                while (paused && IsKeyUp(KEY_SPACE))
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

                BeginDrawing();
                {
                    ClearBackground(backgroundColor);

                    if (IsKeyPressed(KEY_SPACE))
                        paused = !paused;
                    if (IsKeyPressed(KEY_R))
                    {
                        restarted = true;
                        {
                            std::lock_guard lk(m);
                        }
                        cv.notify_one();
                        {
                            std::unique_lock lk(m);
                        }
                    }

                    plotCost();
                    char buffer[64];
                    snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Cost: %f\n", epoch, maxEpoch, n.cost(data));
                    DrawText(buffer, 5, 0, 30, WHITE);
                    drawNetwork();
                }
                EndDrawing();
            }
            CloseWindow();
        }
        void plotCost()
        {
            if (epoch == 0)
                return;

            double rectX = 0;
            double rectY = screenHeight / 4.0;
            double h = screenHeight / 2;
            double w = screenWidth / 2;

            double offX = w / epoch;
            double offY = h / costs[0];
            double lastX = 0;
            double lastY = 0;

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
        void drawNetwork()
        {
            float rectX = screenWidth / 2;
            float rectY = screenHeight / 4;
            float h = screenHeight / 2;
            float w = screenWidth / 2.5;
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
                        double val = n.getWeight(l,a, b);
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
                        double val = n.getBias(i - 1, j);
                        auto sigVal = 1 / (1 + exp(-val));
                        highColor.a = floor(255.f * sigVal);
                        DrawCircle(nodes[i][j].x, nodes[i][j].y, 10, ColorAlphaBlend(lowColor, highColor, WHITE));
                    }
                }
        }
    };

}
