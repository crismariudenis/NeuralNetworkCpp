#pragma once

#include <vector>
#include <cmath>
#include <cassert>
#include <tuple>
#include <functional>
#include <random>

#define printMatrix(x) x.printX(#x);

template <typename T>
class Matrix
{

    size_t rows, cols;

public:
    std::vector<T> data;
    std::tuple<size_t, size_t> shape;

    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols)
    {
        shape = {rows, cols};
        data.resize(cols * rows, T());
    }
    Matrix() : rows(0), cols(0) { shape = {rows, cols}; }
    void rand()
    {

        std::random_device rd{};
        std::mt19937 gen{rd()};

        // init Gaussian distr. w/ N(mean=0, stdev=1/sqrt(rows*cols))
        T n(rows * cols);
        T stdev{1 / sqrt(n)};
        std::normal_distribution<T> d{0, stdev};

        // fill each element w/ draw from distribution
        for (size_t r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
            {
                (*this)(r, c) = d(gen);
            }
    }
    void print()
    {
        for (size_t r = 0; r < rows; r++, std::cout << '\n')
            for (size_t c = 0; c < cols; c++)
                std::cout << (*this)(r, c) << " ";
        std::cout << '\n';
    }
    void printShape()
    {
        std::cout << "Matrix Size([" << rows << ',' << cols << "])\n";
    }

    T &operator()(size_t row, size_t col)
    {
        return data[row * cols + col];
    }

    Matrix &operator*=(Matrix &m)
    {
        assert(cols == m.rows);
        Matrix output{rows, m.cols};
        for (size_t r = 0; r < output.rows; r++)
            for (size_t c = 0; c < output.cols; c++)
                for (size_t k = 0; k < cols; k++)
                    output(r, c) += (*this)(r, k) * m(k, c);
        return (*this = output);
    }
    Matrix operator*(Matrix &m)
    {
        assert(cols == m.rows);
        Matrix output = *this;

        return (output *= m);
    }
    Matrix &operator+=(Matrix &m)
    {
        assert(shape == m.shape);
        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++)
                (*this)(r, c) += m(r, c);
        return *this;
    }
    Matrix operator+(Matrix &m)
    {
        assert(shape == m.shape);
        Matrix output = m;

        return (output += (*this));
    }
    Matrix &operator-=(Matrix &m)
    {
        assert(shape == m.shape);
        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++)
                (*this)(r, c) -= m(r, c);
        return *this;
    }
    Matrix operator-(Matrix &m)
    {
        assert(shape == m.shape);
        Matrix output = m;

        return (output -= (*this));
    }

    Matrix activate(const std::function<T(const T &)> &function)
    {
        Matrix output((*this));
        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++)
                output(r, c) = function((*this)(r, c));
        return output;
    }

    void printX(std::string name = "")
    {
        if (!name.empty())
            std::cout << name << " =";
        std::cout << " [\n";
        for (size_t r = 0; r < rows; r++, std::cout << '\n')
        {
            std::cout << "   ";
            for (size_t c = 0; c < cols; c++)
                std::cout << (*this)(r, c) << " ";
        }
        std::cout << "   ]\n\n";
    }
};
static double random(double min, double max)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<double> d{min, max};
    return d(gen);
}
