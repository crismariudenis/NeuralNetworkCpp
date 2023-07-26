#pragma once
#include <vector>
#include <fstream>

#include "matrix.h"
namespace nn
{
    template <typename T>
    struct DataPoint
    {
        // Todo: Switch to only Matrix objects in Datapoint / vector -> Matrix converter
        // Todo: Find if getOutput function is enough
        std::vector<T> input;
        std::vector<T> output;
    };

    template <typename T>
    class DataSet
    {
    private:
        //  std::ifstream file;
        std::vector<DataPoint<T>> data;

    public:
        DataSet() {}
        DataSet(std::string filename)
        {
            //  file.open(filename.c_str());
        }

        DataSet<T> &operator=(const DataSet<T> &other)
        {
            if (this != &other)
            {
                // Perform a deep copy of the data
                data = other.data;
                // file = other.file;
            }
            return *this;
        }

        size_t size()
        {
            return data.size();
        }
        void addData(std::vector<T> &input, std::vector<T> &output)
        {
            DataPoint<T> dataPoint;
            dataPoint.input = input;
            dataPoint.output = output;
            data.push_back(dataPoint);
        }
        DataPoint<T> &getData(size_t index)
        {
            return data[index];
        }
        void print() const
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
        void generateData()
        {
            const std::vector<std::vector<T>> t = {
                {1, 1, 0},
                {1, 0, 1},
                {0, 1, 1},
                {0, 0, 0},
            };

            for (const auto &x : t)
            {
                std::vector<T> input{x[0], x[1]};
                std::vector<T> output{x[2]};
                this->addData(input, output);
            }
        }
        /// @brief
        /// @param index testcase number
        /// @return
        Matrix<T> getInputMat(size_t index)
        {
            assert(index < data.size());
            Matrix<T> m{1, data[0].input.size()};
            for (size_t i = 0; i < data[index].input.size(); i++)
                m(0, i) = data[index].input[i];
            return m;
        }
        Matrix<T> getOutputMat(size_t index)
        {
            Matrix<T> m{1, data[0].output.size()};
            for (size_t i = 0; i < data[index].output.size(); i++)
                m(0, i) = data[index].output[i];
            return m;
        }
        void testGetInput()
        {
            for (size_t j = 0; j < data.size(); j++, std::cout << '\n')
                for (size_t i = 0; i < data[j].input.size(); i++)
                {

                    std::cout << data[j].input[i];
                }
        }
    };
}
