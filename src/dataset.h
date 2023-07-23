#include <vector>
#include <fstream>

template <typename T>
struct DataPoint
{
    std::vector<T> input;
    std::vector<T> output;
};

template <typename T>
class DataSet
{
private:
    std::ifstream file;

public:
    std::vector<DataPoint<T>> data;
    DataSet(std::string filename)
    {
        file.open(filename.c_str());
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
            {0, 0},
            {1, 2},
            {2, 4},
            {3, 6},
            {4, 8},
        };
        for (const auto &x : t)
        {
            std::vector<T> input{x[0]};
            std::vector<T> output{x[1]};
            this->addData(input, output);
        }
    }
};
