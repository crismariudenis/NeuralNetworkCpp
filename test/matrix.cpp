#include <iostream>

#include "../src/matrix.h"

typedef double T;
int main()
{
    Matrix<T> m{4, 4}, x{4, 4}, n{4, 4};

    m.rand();
    x.rand();
    printMatrix(x);
    x -= m;
    auto y = m - n;
    printMatrix(n);
    printMatrix(m);
    printMatrix(x);
    printMatrix(y);

    x = x.activate([](auto x)
                   { return x / 2.0; });
    printMatrix(x);
    x.printShape();

    return 0;
}