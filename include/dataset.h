#pragma once
#include <vector>
#include <fstream>

#include "matrix.h"
namespace nn
{
    struct DataPoint
    {
        std::vector<T> input;
        std::vector<T> output;
        Matrix getOutputMat()
        {
            Matrix m{1, output.size()};
            for (size_t i = 0; i < output.size(); i++)
                m(0, i) = output[i];
            return m;
        }
        Matrix getInputMat()
        {
            Matrix m{1, input.size()};
            for (size_t i = 0; i < input.size(); i++)
                m(0, i) = input[i];
            return m;
        }
    };

    class DataSet
    {
    private:
        //  std::ifstream file;

    public:
        std::vector<DataPoint> data;
        DataSet();
        DataSet(std::string filename);

        DataSet &operator=(const DataSet &other);

        size_t size();
        void addData(std::vector<T> &input, std::vector<T> &output);
        void shuffle();

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
    void DataSet::shuffle(){
        // std::random_device rd;
        // std::mt19937 g(rd());
        
        std::random_shuffle(data.begin(), data.end());
    }
    Matrix DataSet::getInputMat(size_t index)
    {
        assert(index < data.size());
        Matrix m{1, data[0].input.size()};
        m.data = data[index].input;
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
