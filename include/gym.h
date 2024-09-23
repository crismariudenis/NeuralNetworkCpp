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
    public:
        enum class Mode
        {
            NORMAL = 0,
            UPSCALE,
            TRANSITION
        };

    private:
        class Scroll
        {
            int x = 2 * screenWidth / 3;
            int y = screenHeight - 200;
            int w = 400;
            int h = 10;
            int r = 10;

        public:
            double knobX = 0.5;
            bool dragging = false;

        public:
            void loop()
            {
                DrawRectangle(x, y, w, h, GRAY);
                DrawCircle(x + w * knobX, y + h / 2, r, RED);
                if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) and knobPressed())
                    dragging = true;

                if (dragging)
                { // clamp
                    int newX = std::min(x + w, std::max(x, GetMouseX()));
                    knobX = (float)(newX - x) / float(w);
                }
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
                    dragging = false;
            }
            double getKnobX()
            {
                return knobX;
            }

        private:
            bool knobPressed()
            {
                double d = std::sqrt((x + w * knobX - GetMouseX()) * (x + w * knobX - GetMouseX()) + (y - GetMouseY()) * (y - GetMouseY()));
                return d < ((r / 2) + 10);
            }

        } scroll;

        class ImageTexture
        {
        private:
            int x, y, w, h;
            // int screenWidth, screenHeight;
            const char *path;
            Image img;
            Texture2D tex;

        public:
            ImageTexture(int x, int y, int w, int h, const char *path) : x(x), y(y), w(w), h(h), path(path) {}
            ImageTexture() {}
            void load()
            {
                Texture2D temp;
                img = LoadImage(path); // Load image data into CPU memory (RAM)
                ImageResizeNN(&img, w, h);
                temp = LoadTextureFromImage(img); // Image converted to texture, GPU memory (RAM -> VRAM)

                if (!IsTextureReady(temp))
                    return;

                tex = temp;
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
        static const int scale = 80;
        static const int screenWidth = 16 * scale;
        static const int screenHeight = 9 * scale;
        const char *windowName = "Neural Network";
        const char *outputPath = "./data/out.png";
        // For the digit recognition
        const int imageWidth = 200;
        const int imageHeight = 200;
        int nrWindows = 2;
        std::vector<ImageTexture> imgs;
        ImageTexture out, in;
        Mode mode = Mode::NORMAL;

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
        Gym(nn::NeuralNetwork &n, Mode mode);

        void setup();
        void train(const nn::DataSet &ds, size_t epochs);
        void computing();
        void drawing();
        void plotCost();
        void drawNetwork();
        void upscale();
        void transition();
    };
    Gym::Gym(nn::NeuralNetwork &n, Mode mode = Mode::NORMAL) : n(n), mode(mode)
    {
        setup();
    }
    void Gym::train(const nn::DataSet &ds, size_t epochs)
    {
        this->epochs = epochs;
        this->ds = ds;
        if (epochs <= 0)
            return;
        costs.resize(epochs);

        // thread computing cause drawing is slow
        std::thread t1(&Gym::computing, this);
        std::thread t2;
        switch (mode)
        {
        case Mode::UPSCALE:
            t2 = std::thread(&Gym::upscale, this);
            break;
        case Mode::TRANSITION:
            t2 = std::thread(&Gym::transition, this);
            break;
        case Mode::NORMAL:
            break;
        }
        drawing();

        if (t2.joinable())
            t2.join();
        if (t1.joinable())
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
            while (paused && IsKeyUp(KEY_SPACE) /*|| n.isRandomizing*/)
            {
            }
            costs[epoch] = n.cost(ds);
            n.train(ds, epoch);
        }

#ifdef TIME
        exit(0);
#endif
    }
    void Gym::setup()
    {
        InitWindow(screenWidth, screenHeight, windowName);
        SetTargetFPS(60);

        int pad = 20;
        switch (mode)
        {
        case Mode::UPSCALE:
            nrWindows = 3;
            imgs.reserve(2);
            imgs[0] = ImageTexture{screenWidth - 2 * imageWidth - pad, 2 * screenHeight / 3 - imageHeight, imageWidth, imageHeight, "./data/9.png"};
            imgs[0].load();
            imgs[1] = ImageTexture{screenWidth - imageWidth - pad, 2 * screenHeight / 3 - imageHeight, imageWidth, imageHeight, outputPath};
            imgs[1].load();

            break;
        case Mode::TRANSITION:
            nrWindows = 3;
            imgs.reserve(3);
            imgs[0] = ImageTexture{2 * screenWidth / 3, 2 * screenHeight / 3 - imageHeight, imageWidth, imageHeight, "./data/9.png"};
            imgs[0].load();
            imgs[1] = ImageTexture{2 * screenWidth / 3 + imageWidth, 2 * screenHeight / 3 - imageHeight, imageWidth, imageHeight, "./data/8.png"};
            imgs[1].load();
            imgs[2] = ImageTexture{screenWidth - 3 * imageWidth / 2 - pad, 2 * screenHeight / 3 - 2 * imageHeight, imageWidth, imageHeight, outputPath};
            imgs[2].load();
            break;
        case Mode::NORMAL:
            break;
        }
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
                if (IsKeyPressed(KEY_RIGHT))
                    n.rate *= 10;
                if (IsKeyPressed(KEY_LEFT))
                    n.rate *= 0.1;
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
                char buffer[100];
                snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Momentum: %f, Cost: %f\n", epoch, epochs, n.rate, n.momentum, n.lastCost);
                DrawText(buffer, 5, 0, 30, WHITE);

                drawNetwork();
                switch (mode)
                {
                case Mode::UPSCALE:
                    imgs[1].load();
                    imgs[0].draw();
                    imgs[1].draw();
                    break;
                case Mode::TRANSITION:
                    imgs[0].draw();
                    imgs[1].draw();
                    imgs[2].load();
                    imgs[2].draw();
                    scroll.loop();
                    break;
                case Mode::NORMAL:
                    break;
                }
            }
            EndDrawing();
        }
        for (auto &x : imgs)
            x.unload();
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
                    DrawCircleV(nodes[i][j], 10, GRAY);
                else
                {
                    T val = n.getBias(i - 1, j);
                    auto sigVal = 1 / (1 + exp(-val));
                    highColor.a = floor(255.f * sigVal);
                    DrawCircleV(nodes[i][j], 10, ColorAlphaBlend(lowColor, highColor, WHITE));
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
    void Gym::upscale()
    {
        int nr = 0;
        NeuralNetwork m{n.getArch()};
        while (!closed)
        {
            if (nr % 10 == 0)
            {
                m.copyWeightsAndBiasesFrom(n);

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
            nr %= 10;
            nr++;
        }
    }
    void Gym::transition()
    {
        int nr = 0;
        NeuralNetwork m{n.getArch()};
        while (!closed)
        {
            if (nr % 10 == 0)
            {
                size_t imageHeight = 28;
                size_t imageWidth = 28;
                m.copyWeightsAndBiasesFrom(n);

                uint8_t *outPixels = (uint8_t *)malloc(sizeof(*outPixels) * imageHeight * imageWidth);
                for (int y = 0; y < imageHeight; y++)
                    for (int x = 0; x < imageWidth; x++)
                    {
                        nn::Matrix input{1, 3};
                        input(0, 0) = nn::T(x) / (imageWidth - 1.0);
                        input(0, 1) = nn::T(y) / (imageHeight - 1.0);
                        input(0, 2) = scroll.getKnobX();
                        nn::Matrix output = m.forward(input);

                        uint8_t pixel = (uint8_t)(output(0, 0) * 255.0);
                        outPixels[y * imageHeight + x] = pixel;
                    }
                stbi_write_png(outputPath, imageWidth, imageHeight, 1, outPixels, imageWidth * (sizeof(*outPixels)));
                free(outPixels);
            }
            nr %= 10;
            nr++;
        }
    }
}
