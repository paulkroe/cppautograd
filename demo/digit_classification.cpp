#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include "../grad/cppgrad.h"
#include "../grad/modules.h"
#include "../performance_evals/timer.h"

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
                /* Skip empty lines */
                continue;
            }

            std::stringstream ss(line);
            std::string value;

            /* Parse label (first value in the row) */
            std::getline(ss, value, ',');
            /* Store labels as float */
            float label = std::stof(value);

            /* Parse pixels (remaining 784 values) */
            std::vector<float> pixels;
            while (std::getline(ss, value, ',')) {
                if (!value.empty()) {
                    /* Normalize to [0, 1] */
                    pixels.push_back(std::stof(value) / 255.0f);
                }
            }

            if (pixels.size() != 784) {
                throw std::runtime_error("Invalid row: expected 784 pixel values, got " +
                                         std::to_string(pixels.size()));
            }

            /* Store as Tensors */
            images_.push_back(torch::tensor(pixels).view({1, 28, 28}));
            /* Store labels as float */
            labels_.push_back(torch::tensor(label, torch::kFloat32));
        }
    }

    torch::data::Example<> get(size_t index) override {
        return {images_[index], labels_[index]};
    }

    torch::optional<size_t> size() const override {
        return images_.size();
    }
};

void train(const size_t num_threads = 1) {
    const std::string train_path = "../../demo/data/archive/mnist_train.csv";
    const std::string test_path = "../../demo/data/archive/mnist_test.csv";
    
    /* Define the model */
    Linear linear1(784, 512);
    Linear linear2(512, 256);
    Linear linear3(256, 128);
    Linear linear4(128, 64);
    Linear linear5(64, 10);

    try {
        /* Create Dataset and DataLoader */
        auto options = torch::data::DataLoaderOptions().batch_size(16);
        
        auto dataset_train = MNISTDataset(train_path)
                           .map(torch::data::transforms::Normalize<>(0.0, 1.0)) // Normalize
                           .map(torch::data::transforms::Stack<>());

        auto data_loader_train = torch::data::make_data_loader(
            std::move(dataset_train), options);

        auto dataset_test = MNISTDataset(test_path)
                           .map(torch::data::transforms::Normalize<>(0.0, 1.0)) // Normalize
                           .map(torch::data::transforms::Stack<>());

        auto data_loader_test = torch::data::make_data_loader(
            std::move(dataset_test), options);

        int epoch = 0;
        
        /* TRAINING LOOP */
        int num_epochs = 1;

        /* Compute the total number of batches in the DataLoader */
        int num_batches_train = std::distance(data_loader_train->begin(), data_loader_train->end());

        for (int epoch = 0; epoch < num_epochs; epoch++) {
            /* Current batch number */
            int batch_idx = 0;

            for (auto& batch : *data_loader_train) {
                /* Shape: [batch_size, 1, 28, 28] */
                auto images = batch.data;
                /* Shape: [batch_size] */
                auto labels = batch.target;

                /* Flatten images to [batch_size, 784] */
                auto flattened_images = images.view({images.size(0), -1}).to(torch::kCPU).contiguous();

                /* Convert images to std::vector<float> */
                std::vector<float> images_vec(flattened_images.numel());
                std::memcpy(
                    images_vec.data(),
                    flattened_images.data_ptr<float>(),
                    images_vec.size() * sizeof(float)
                );

                /* Convert labels to std::vector<float> (stored as float instead of int) */
                std::vector<float> labels_vec(labels.numel());
                std::memcpy(
                    labels_vec.data(),
                    labels.data_ptr<float>(),
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

                /* Print loss with epoch and batch progress */
                if (batch_idx % 500 == 0) {
                std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                        << "Batch [" << (batch_idx + 1) << "/" << num_batches_train << "] "
                        << "Loss: " << loss.data[0] << std::endl;
                }

                /* Update weights */
                for (auto layer : {&linear1, &linear2, &linear3, &linear4, &linear5}) {
                    for (size_t i = 0; i < layer->weight.data.size(); i++) {
                        layer->weight.data[i] -= 0.1 * layer->weight.grad().data[i];
                    }
                    for (size_t i = 0; i < layer->bias.data.size(); i++) {
                        layer->bias.data[i] -= 0.1 * layer->bias.grad().data[i];
                    }
                    layer->weight.zero_grad();
                    layer->bias.zero_grad();
                }

                batch_idx++;
            }
        }


        /* EVALUATION PHASE */
        for (auto layer : {&linear1, &linear2, &linear3, &linear4, &linear5}) {
            layer->eval();
        }

        int total_correct = 0;
        int total_samples = 0;
        std::unordered_map<int, int> correct_per_class;
        std::unordered_map<int, int> total_per_class;
        
        /* Compute the total number of batches in the DataLoader */
        int num_batches_test = std::distance(data_loader_test->begin(), data_loader_test->end());
        
        for (auto& batch : *data_loader_test) {
            auto images = batch.data;
            auto labels = batch.target;

            auto flattened_images = images.view({images.size(0), -1}).to(torch::kCPU).contiguous();

            std::vector<float> images_vec(flattened_images.numel());
            std::memcpy(
                images_vec.data(),
                flattened_images.data_ptr<float>(),
                images_vec.size() * sizeof(float)
            );

            std::vector<float> labels_vec(labels.numel());
            std::memcpy(
                labels_vec.data(),
                labels.data_ptr<float>(),
                labels_vec.size() * sizeof(float)
            );

            /* Forward pass without gradient calculation */
            Tensor x = Tensor(images_vec, {16, 784}, false);
            Tensor y_pred = linear5.forward(
                                        linear4.forward(
                                                linear3.forward(
                                                        linear2.forward(
                                                                linear1.forward(x).relu()
                                                        ).relu()
                                                ).relu()
                                        )
                                );

            std::vector<float> predictions(y_pred.data.begin(), y_pred.data.end());

            /* Convert softmax output to class prediction */
            for (size_t i = 0; i < labels_vec.size(); i++) {
                int predicted_class = std::max_element(predictions.begin() + i * 10,
                                                       predictions.begin() + (i + 1) * 10) - (predictions.begin() + i * 10);
                int true_class = static_cast<int>(labels_vec[i]);

                if (predicted_class == true_class) {
                    total_correct++;
                    correct_per_class[true_class]++;
                }

                total_per_class[true_class]++;
                total_samples++;
            }
        }

        // Compute accuracies
        float overall_accuracy = 100.0f * total_correct / total_samples;
        std::cout << "Overall Accuracy: " << overall_accuracy << "%" << std::endl;

        for (const auto& [class_label, correct] : correct_per_class) {
            float class_accuracy = 100.0f * correct / total_per_class[class_label];
            std::cout << "Accuracy for class " << class_label << ": " << class_accuracy << "%" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return;
}

int main() {
    train();
    float time = 0.0f;
    std::vector<int> num_threads = {1, 2, 4, 8};
    
    for (auto t: num_threads) {
        time += ExecutionTimer::measure("train", [t]() {      train(t);    });
        std::cout << "=========================================================" << std::endl;
        std::cout << "Average time for training with " << t << " threads: " << time << " ms" << std::endl;
        time = 0.0f;
    }
}