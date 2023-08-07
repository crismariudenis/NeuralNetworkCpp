#pragma once
#include <vector>
#include <fstream>

#include "matrix.h"
namespace nn
{
    typedef double T;
    struct DataPoint
    {
        std::vector<T> input;
        std::vector<T> output;
    };

    class DataSet
    {
    private:
        //  std::ifstream file;
        std::vector<DataPoint> data;

    public:
        DataSet();
        DataSet(std::string filename);

        DataSet &operator=(const DataSet &other);

        size_t size();
        void addData(std::vector<T> &input, std::vector<T> &output);
        void generateData();


        DataPoint &getData(size_t index);
        Matrix getInputMat(size_t index);
        Matrix getOutputMat(size_t index);

        void print() const;
    };
    DataSet::DataSet() {}
    DataSet::DataSet(std::string filename)
    {
        //  file.open(filename.c_str());
    }
    DataSet &DataSet::operator=(const DataSet &other)
    {
        if (this != &other)
        {
            // Perform a deep copy of the data
            data = other.data;
            // file = other.file;
        }
        return *this;
    }
    size_t DataSet::size()
    {
        return data.size();
    }
    void DataSet::addData(std::vector<T> &input, std::vector<T> &output)
    {
        DataPoint dataPoint;
        dataPoint.input = input;
        dataPoint.output = output;
        data.push_back(dataPoint);
    }
    DataPoint &DataSet::getData(size_t index)
    {
        return data[index];
    }
    void DataSet::print() const
    {
        for (size_t i = 0; i < data.size(); i++)
        {
            std::cout << "Input: ";
            for (const auto &x : data[i].input)
                std::cout << x << " ";
            std::cout << "Output: ";
            for (const auto &x : data[i].output)
                std::cout << x << " ";
            std::cout << '\n';
        }
    }
    void DataSet::generateData()
    {
        const std::vector<std::vector<T>> t = {
            {1, 1, 0},
            {1, 0, 1},
            {0, 1, 1},
            {0, 0, 0},
        };

        // for (const auto &x : t)
        // {
        //     std::vector<T> input{x[0], x[1]};
        //     std::vector<T> output{x[2]};
        //     this->addData(input, output);
        // }
        for (int i = 0; i < 10; i++)
        {
            int a = rand() % 8;
            int b = rand() % 8;
            std::vector<T> input;
            // add a and b 's bits

            for (int j = 0; j < 4; j++)
                input.push_back((a >> j) & 1);

            for (int j = 0; j < 4; j++)
                input.push_back((b >> j) & 1);

            std::vector<T> output;
            // add a + b 's bits
            for (int j = 0; j < 5; j++)
                output.push_back(((a + b) >> j) & 1);
            this->addData(input, output);
        }
    }
    Matrix DataSet::getInputMat(size_t index)
    {
        assert(index < data.size());
        Matrix m{1, data[0].input.size()};
        for (size_t i = 0; i < data[index].input.size(); i++)
            m(0, i) = data[index].input[i];
        return m;
    }
    Matrix DataSet::getOutputMat(size_t index)
    {
        Matrix m{1, data[0].output.size()};
        for (size_t i = 0; i < data[index].output.size(); i++)
            m(0, i) = data[index].output[i];
        return m;
    }

}
