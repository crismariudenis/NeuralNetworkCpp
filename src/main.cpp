#include <iostream>

#include "matrix.h"

int main()
{
    typedef float T;
    Matrix<T> m{4, 4}, n{4, 4};
    m.init(), n.init();
    Matrix<T> x;
    Matrix<T> y;

    x = m * n;
    y = m + n;

    printMatrix(n);
    printMatrix(m);
    printMatrix(x);
    printMatrix(y);

    x.printShape();
    return 0;
}