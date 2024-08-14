#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include <vector>
#include <thread>
#include <torch/script.h>
#include <torch/torch.h>

class ModelRunner {
private:
    torch::jit::script::Module model;
    std::string filename;
    int feature_dim;
    std::deque<torch::Tensor> buffer;
    size_t capacity;
    torch::Device device; // Store the device type
    int cnt = 0;

public:
    ModelRunner(const std::string& model_path, const std::string& filename, int feature_dim, size_t capacity)
        : filename(filename), feature_dim(feature_dim), capacity(capacity), 
         device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
        try {
            model = torch::jit::load(model_path);
            // Set device based on CUDA availability
            model.to(device);
            std::cout << (device.type() == torch::kCUDA ? "Using GPU." : "Using CPU.") << std::endl;

            // Verify model input dimension with a random input
            auto test_input = torch::rand({1, feature_dim}, device);
            std::vector<torch::jit::IValue> inputs = {test_input};
            model.forward(inputs); // This will throw if the dimension is incorrect
        } catch (const c10::Error& err) {
            std::cerr << "Error loading the model: " << err.what() << std::endl;
            exit(-1);
        }
    }

    torch::Tensor convertLineToTensor(const std::string& line) {
        torch::Tensor tensor = torch::empty(feature_dim, torch::kFloat32);
        std::stringstream ss(line);
        std::string item;
        int index = 0;
        while (std::getline(ss, item, ',') && index < feature_dim) {
            try {
                tensor[index++] = std::stof(item);
            } catch (const std::exception& e) {
                std::cerr << "Parsing error: " << e.what() << " in line: " << line << std::endl;
                tensor[index++] = 0.0f; // Handle error, e.g., by setting to zero
            }
        }
        return tensor;
    }

    void feedModel(){
        if (buffer.empty()) return;

        // Stack tensors to create a batched input
        auto input = torch::stack(std::vector<torch::Tensor>(buffer.begin(), buffer.end()));

        // Forward pass through the model
        auto outputs = model.forward({input}).toTuple();
        auto predictions = outputs->elements()[0].toTensor();
        std::cout << "Model output " << this->cnt + 1 << ": " 
                  << std::setw(3) << std::setfill(' ');
        this->cnt++;
        for (auto s : predictions.sizes()) std::cout << s << " ";
        std::cout << std::endl;
    }

    void updateBuffer(torch::Tensor newTensor) {
        if (buffer.size() >= capacity) {
            buffer.pop_front();  // Remove the oldest tensor if we are at capacity
        }
        buffer.push_back(newTensor);  // Add the new tensor to the buffer
    }

    void processFile(const std::string& spec_filename) {
        auto start = std::chrono::high_resolution_clock::now();

        std::string effectiveFilename = spec_filename.empty() ? this->filename : spec_filename;
        std::cout << "Processing file: " << effectiveFilename << std::endl;
        std::ifstream file(effectiveFilename);
        std::string line;
        if (std::getline(file, line)) {} // Optionally handle header

        while (std::getline(file, line)) {
            auto tensor = convertLineToTensor(line);
            updateBuffer(tensor);  // Update the buffer with each new line
            feedModel();           // Run the model on every new line
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout <<"\n\n\n";
        std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;
        std::cout << "Processed " << this->cnt << " lines." << std::endl;
        std::cout << "Speed: " << this->cnt / elapsed.count() << " lines per second." << std::endl;
    }


    void start(const std::string& filename = "") {
        this->cnt = 0;
        std::thread worker(&ModelRunner::processFile, this, filename);
        worker.join();
    }
};

int main() {
    ModelRunner runner("/home/shaoze/Documents/Boeing/Boeing-Trajectory-Prediction/exported/model_tft_vqvae_cpu.pt",
                       "/home/shaoze/Documents/Boeing/Boeing-Trajectory-Prediction/pipeline/demo/0.csv",
                       32, // Feature dimension
                       40); // Capacity or batch size

    runner.start("/home/shaoze/Documents/Boeing/Boeing-Trajectory-Prediction/pipeline/demo/1.csv");
    return 0;
}
