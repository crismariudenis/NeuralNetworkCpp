#pragma once

#include <vector>
#include <cmath>
#include <cassert>
#include <tuple>
#include <functional>
#include <random>

#define printMatrix(x) x.printX(#x);

template <typename Type>
class Matrix
{

    size_t cols, rows;

public:
    std::vector<Type> data;
    std::tuple<size_t, size_t> shape;

    Matrix(size_t rows, size_t cols) : cols(cols), rows(rows)
    {
        shape = {rows, cols};
        data.resize(cols * rows, Type());
    }
    Matrix() : cols(0), rows(0) { shape = {rows, cols}; }

    void rand()
    {

        std::random_device rd{};
        std::mt19937 gen{rd()};

        // init Gaussian distr. w/ N(mean=0, stdev=1/sqrt(rows*cols))
        Type n(rows*cols);
        Type stdev{1 / sqrt(n)};
        std::normal_distribution<Type> d{0, stdev};

        // fill each element w/ draw from distribution
        for (size_t r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                (*this)(r, c) = d(gen);
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

    Type &operator()(size_t row, size_t col)
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

    Matrix activate(const std::function<Type(const Type &)> &function)
    {
        Matrix output((*this));
        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++)
                output(r, c) = function((*this)(r, c));
        return output;
    }

    // Only called with the printMatrix() macro
    void printX(std::string name="")
    {
        if(!name.empty())
        std::cout << name<<" =";
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
