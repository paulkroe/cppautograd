#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <vector>
#include "../vanilla/cppgrad.h"
#include "../vanilla/modules.h"

class MNISTDataset : public torch::data::datasets::Dataset<MNISTDataset> {
    std::vector<torch::Tensor> images_;
    std::vector<torch::Tensor> labels_;

public:
    explicit MNISTDataset(const std::string& csv_path) {
        std::ifstream file(csv_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + csv_path);
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue; // Skip empty lines
            }

            std::stringstream ss(line);
            std::string value;

            // Parse label (first value in the row)
            std::getline(ss, value, ',');
            int label = std::stoi(value);

            // Parse pixels (remaining 784 values)
            std::vector<float> pixels;
            while (std::getline(ss, value, ',')) {
                if (!value.empty()) {
                    pixels.push_back(std::stof(value) / 255.0f); // Normalize to [0, 1]
                }
            }

            if (pixels.size() != 784) {
                throw std::runtime_error("Invalid row: expected 784 pixel values, got " +
                                         std::to_string(pixels.size()));
            }

            // Store as Tensors
            images_.push_back(torch::tensor(pixels).view({1, 28, 28}));
            labels_.push_back(torch::tensor(static_cast<float>(label), torch::kFloat32)); // Store labels as float
        }
    }

    torch::data::Example<> get(size_t index) override {
        return {images_[index], labels_[index]};
    }

    torch::optional<size_t> size() const override {
        return images_.size();
    }
};

int main() {
    const std::string dataset_path = "../../tests/data/archive/mnist_train.csv";
    // todo add acitvation function
    Linear linear1(784, 512);
    Linear linear2(512, 256);
    Linear linear3(256, 128);
    Linear linear4(128, 64);
    Linear linear5(64, 10);


    try {
        // Create Dataset and DataLoader
        auto dataset = MNISTDataset(dataset_path)
                           .map(torch::data::transforms::Normalize<>(0.0, 1.0)) // Normalize
                           .map(torch::data::transforms::Stack<>());

        auto options = torch::data::DataLoaderOptions().batch_size(16);

        auto data_loader = torch::data::make_data_loader(
            std::move(dataset), options);

        int i = 0;
        // Iterate through the DataLoader
        for (auto& batch : *data_loader) {
            auto images = batch.data;  // Shape: [batch_size, 1, 28, 28]
            auto labels = batch.target; // Shape: [batch_size]

            // Flatten images to [batch_size, 784]
            auto flattened_images = images.view({images.size(0), -1}).to(torch::kCPU).contiguous();

            // Ensure labels are contiguous
            auto labels_contiguous = labels.to(torch::kCPU).to(torch::kFloat32).contiguous();

            // Convert images to std::vector<float>
            std::vector<float> images_vec(flattened_images.numel());
            std::memcpy(
                images_vec.data(),
                flattened_images.data_ptr<float>(),
                images_vec.size() * sizeof(float)
            );

            // Convert labels to std::vector<float>
            std::vector<float> labels_vec(labels.numel());
            std::memcpy(
                labels_vec.data(),
                labels_contiguous.data_ptr<float>(),
                labels_vec.size() * sizeof(float)
            );

            /* Perform forward pass */
            Tensor x = Tensor(images_vec, {16, 784}, false);
            Tensor y = Tensor(labels_vec, {16}, false);
            Tensor y_pred = linear5.forward(
                                        linear4.forward(
                                                linear3.forward(
                                                        linear2.forward(
                                                                linear1.forward(x).relu()
                                                        ).relu()
                                                ).relu()
                                        )
                                );
             
            Tensor loss = CrossEntropyLoss(y_pred, y);
            loss.backward();

            // std::cout << "y_pred: " << y_pred << std::endl;
            std::cout << "Epoch " << i++ << std::endl;
            std::cout << "Loss: " << loss.data[0] << std::endl;
            
            /* Update weights */
            for (auto layer : {&linear1, &linear2, &linear3}) {
                for (size_t i = 0; i < layer->weight.data.size(); i++) {
                    layer->weight.data[i] -= 0.1 * layer->weight.grad->data[i];
                }
                for (size_t i = 0; i < layer->bias.data.size(); i++) {
                    layer->bias.data[i] -= 0.1 * layer->bias.grad->data[i];
                }
                layer->weight.zero_grad();
                layer->bias.zero_grad();
            }

        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}