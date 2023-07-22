#include <vector>
#include <fstream>

template <typename T>
struct DataPoint
{
    std::vector<T> input;
    std::vector<T> output;
};

template <typename T>
class Data
{
private:
    std::ifstream file;

public:
    std::vector<DataPoint<T>> data;
    Data(std::string filename)
    {
        file.open(filename.c_str());
    }
    size_t size()
    {
        return data.size();
    }
    void addData(std::vector<T> input, std::vector<T> output)
    {
        DataPoint<T> dataPoint;
        dataPoint.input = input;
        dataPoint.output = output;
        data.push_back(dataPoint);
    }
    DataPoint<T> &getData(int index)
    {
        return data[index];
    }
    void print()
    {
        for (size_t i = 0; i < data.size(); i++)
        {
            std::cout << "Input: ";
            for (auto &x : data[i].input)
                std::cout << x << " ";
            std::cout << "Output: ";
            for (auto &x : data[i].output)
                std::cout << x << " ";
            std::cout << '\n';
        }
    }
};
