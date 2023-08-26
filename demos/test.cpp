#include <iostream>
#include "../include/raylib.h"
int main()
{
    const int screenWidth = 800;
    const int screenHeight = 450;

    InitWindow(screenWidth, screenHeight, "suii");

    Image image = LoadImage("./data/9.png"); // Load image data into CPU memory (RAM)

    Color *colors = LoadImageColors(image);

    std::cout << image.height << '\n';
    for (int i = 0; i < image.height; i++, std::cout << '\n')
        for (int j = 0; j < image.width; j++)
        {
            int index = (i * image.width) + j;
            // std::cout << (bool)colors[index].r << " ";
            if ((bool)colors[index].r)
                std::cout << char(254);
            else
                std::cout << " ";

            std::cout << " ";

        }
    ImageResize(&image, 200, 200);
    Texture2D texture = LoadTextureFromImage(image); // Image converted to texture, GPU memory (RAM -> VRAM)
    UnloadImage(image);                              // Unload image data from CPU memory (RAM)

    image = LoadImageFromTexture(texture); // Load image from GPU texture (VRAM -> RAM)
    UnloadTexture(texture);                // Unload texture from GPU memory (VRAM)

    texture = LoadTextureFromImage(image); // Recreate texture from retrieved image data (RAM -> VRAM)
    UnloadImage(image);

    while (!WindowShouldClose())
    {

        BeginDrawing();
        {
            ClearBackground(RAYWHITE);

            DrawTexture(texture, screenWidth / 2 - texture.width / 2, screenHeight / 2 - texture.height / 2, WHITE);

            DrawText("this IS a texture loaded from an image!", 300, 370, 10, GRAY);
        }
        EndDrawing();
    }

    UnloadTexture(texture); // Texture unloading

    CloseWindow();
}