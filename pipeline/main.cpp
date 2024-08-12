#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include "libtorch/include/torch/script.h"


class RollingFIFO {
private:
    std::deque<std::string> buffer;
    size_t capacity;

public:
    RollingFIFO(size_t n) : capacity(n) {}

    void push(const std::string& line) {
        buffer.push_back(line);
        if (buffer.size() > capacity) {
            buffer.pop_front();
        }
    }

    std::deque<std::string> getLastN() const {
        return buffer;
    }

    bool isFull() const {
        return buffer.size() == capacity;
    }
};

// torch::Tensor convertLineToTensor(const std::string& line) {
//     std::vector<float> values;
//     std::stringstream ss(line);
//     std::string item;
//     while (std::getline(ss, item, ',')) {
//         values.push_back(std::stof(item));
//     }
//     return torch::tensor(values);
// }

torch::Tensor convertLineToTensor(const std::string& line) {
    std::vector<float> values;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            values.push_back(std::stof(item));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << item << " in line: " << line << std::endl;
            continue; // Skip invalid entries or handle them appropriately
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range: " << item << " in line: " << line << std::endl;
            continue;
        }
    }
    return torch::tensor(values);
}

void feedModel(torch::jit::script::Module& model, const RollingFIFO& fifo) {
    auto data = fifo.getLastN();
    std::vector<torch::Tensor> tensors;
    for (const auto& line : data) {
        tensors.push_back(convertLineToTensor(line));
    }
    torch::Tensor input = torch::stack(tensors);
    auto output = model.forward({input}).toTensor();
    std::cout << "Model output: " << output << std::endl;
}

void readCSVAndProcess(const std::string& filename, RollingFIFO& fifo, torch::jit::script::Module& model) {
    std::ifstream file(filename);
    std::string line;
    
    // remove header
    if (std::getline(file, line)) {
        // Optionally do something with the header or just ignore it
    }

    while (std::getline(file, line)) {
        fifo.push(line);
        if (fifo.isFull()) {
            feedModel(model, fifo);
        }
    }
    // Optional: feed model with remaining data if necessary
    if (!fifo.getLastN().empty()) {
        feedModel(model, fifo);
    }
}

int main() {
    try {
        torch::jit::script::Module model = torch::jit::load("/home/shaoze/Documents/Boeing/Boeing-Trajectory-Prediction/exported/model_tft_vqvae_cpu.pt");
        RollingFIFO fifo(40); // Last 5 rows

        readCSVAndProcess("/home/shaoze/Documents/Boeing/Boeing-Trajectory-Prediction/pipeline/demo/0.csv", fifo, model);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
