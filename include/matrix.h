#pragma once

#include <vector>
#include <cmath>
#include <cassert>
#include <tuple>
#include <functional>
#include <random>

#define printMatrix(x) x.printX(#x);
namespace nn
{
    typedef double T;
    class Matrix
    {

        size_t rows, cols;

    public:
        std::vector<T> data;
        std::tuple<size_t, size_t> shape;

        Matrix(size_t rows, size_t cols);
        Matrix();
        void rand();
        void fill(T value);
        Matrix &activate(const std::function<T(const T &)> &function);

        T &operator()(size_t row, size_t col);
        inline Matrix &operator*=(T x);
        inline Matrix &operator*(T x);
        inline Matrix &operator*=(const Matrix &m);
        inline Matrix &operator*(const Matrix &m);
        inline Matrix &operator+=(const Matrix &m);
        inline Matrix &operator+(const Matrix &m);
        inline Matrix &operator-=(const Matrix &m);
        inline Matrix &operator-(const Matrix &m);

        void print();
        void printShape();
        void printX(std::string name);
    };

    Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols)
    {
        shape = {rows, cols};
        data.resize(cols * rows, T());
    }
    Matrix::Matrix() : rows(0), cols(0) { shape = {rows, cols}; }

    void Matrix::rand()
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
    void Matrix::fill(T value)
    {
        std::fill(data.begin(), data.end(), value);
    }
    Matrix &Matrix::activate(const std::function<T(const T &)> &function)
    {
        std::transform(this->data.begin(), this->data.end(), this->data.begin(), function);
        return *this;
    }

    T &Matrix::operator()(size_t row, size_t col)
    {
        assert(row < rows);
        assert(col < cols);
        return data[row * cols + col];
    }
    Matrix &Matrix::operator*=(T x)
    {
        std::transform(this->data.begin(), this->data.end(), this->data.begin(),
                       [x](T elem)
                       { return elem * x; });
        return *this;
    }
    Matrix &Matrix::operator*(T x)
    {
        std::transform(this->data.begin(), this->data.end(), this->data.begin(),
                       [x](T elem)
                       { return elem * x; });
        return *this;
    }
    Matrix &Matrix::operator*=(const Matrix &m)
    {
        assert(cols == m.rows);
        std::vector<T> output_data(rows * m.cols);

        for (size_t r = 0; r < rows; r++)
            for (size_t k = 0; k < cols; k++)
                for (size_t c = 0; c < m.cols; c++)
                    output_data[r * m.cols + c] += data[r * cols + k] * m.data[k * m.cols + c];

        cols = m.cols;
        shape = {rows, cols};
        data.swap(output_data);
        return *this;
    }
    Matrix &Matrix::operator*(const Matrix &m)
    {
        if (cols != m.rows)
            std::cout << "cols: {" << cols << "}, m.rows: {" << m.rows << "}\n";

        assert(cols == m.rows);
        (*this) *= m;

        return *this;
    }
    Matrix &Matrix::operator+=(const Matrix &m)
    {
        assert(shape == m.shape);
        std::transform(this->data.begin(), this->data.end(), m.data.begin(), this->data.begin(), std::plus<T>());
        return *this;
    }
    Matrix &Matrix::operator+(const Matrix &m)
    {
        assert(shape == m.shape);
        std::transform(this->data.begin(), this->data.end(), m.data.begin(), this->data.begin(), std::plus<T>());
        return *this;
    }
    Matrix &Matrix::operator-=(const Matrix &m)
    {
        assert(shape == m.shape);
        std::transform(this->data.begin(), this->data.end(), m.data.begin(), this->data.begin(), std::minus<T>());
        return *this;
    }
    Matrix &Matrix::operator-(const Matrix &m)
    {
        assert(shape == m.shape);
        std::transform(this->data.begin(), this->data.end(), m.data.begin(), this->data.begin(), std::minus<T>());
        return *this;
    }
    void Matrix::printX(std::string name = "")
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
    void Matrix::print()
    {
        for (size_t r = 0; r < rows; r++, std::cout << '\n')
            for (size_t c = 0; c < cols; c++)
                std::cout << (*this)(r, c) << " ";
        std::cout << '\n';
    }
    void Matrix::printShape()
    {
        std::cout << "Matrix Size([" << rows << ',' << cols << "])\n";
    }

    static double random(double min, double max)
    {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<double> d{min, max};
        return d(gen);
    }
}