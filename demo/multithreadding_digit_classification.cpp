#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include "../grad/cppgrad.h"
#include "../grad/modules.h"
#include "../performance_evals/timer.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <numeric>
#include <chrono>

#define TIME_IT(code_block, label) {\
    auto start = std::chrono::high_resolution_clock::now(); \
    code_block \
    auto end = std::chrono::high_resolution_clock::now(); \
    std::cout << label << " took: " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() \
              << " ms" << std::endl; \
}

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

std::mutex grad_mutex;
std::vector<float> losses;

void worker(Tensor x, Tensor y, Linear &linear1, Linear &linear2, Linear &linear3, 
            Linear &linear4, Linear &linear5, size_t num_threads) {
    Tensor loss = Tensor({0}, {1}, false);
    TIME_IT({
        Tensor y_pred = linear5.forward(
                                linear4.forward(
                                        linear3.forward(
                                                linear2.forward(
                                                        linear1.forward(x).relu()
                                                ).relu()
                                        ).relu()
                                )
                        );

        loss = CrossEntropyLoss(y_pred, y);
        loss.backward();
    }, "Worker Forward + Backward Pass");

    std::lock_guard<std::mutex> lock(grad_mutex);
    losses.push_back(loss.data()[0]);

    TIME_IT({
        /* Update weights */
        for (auto layer : {&linear1, &linear2, &linear3, &linear4, &linear5}) {
            auto& data_weight = layer->weight.ptr->data;
            auto& grad_weight = layer->weight.grad().ptr->data;

            for (size_t i = 0; i < grad_weight.size(); i++) {
                data_weight[i] -= 0.1 * grad_weight[i]; // Modify real tensor
            }

            auto& data_bias = layer->bias.ptr->data;
            auto& grad_bias = layer->bias.grad().ptr->data;

            for (size_t i = 0; i < grad_bias.size(); i++) {
                data_bias[i] -= 0.1 * grad_bias[i];
            }

            layer->weight.zero_grad();
            layer->bias.zero_grad();
        }   

    }, "Weight Update");
}

void train(const size_t num_threads = 1, const size_t batch_size = 16) {
    //TODO
    const std::string train_path = "../../demo/data/archive/mnist_train.csv";
    // const std::string train_path = "../../demo/data/archive/mnist_test.csv";
    const std::string test_path = "../../demo/data/archive/mnist_test.csv";

    /* Define the model */
    Linear linear1(784, 512);
    Linear linear2(512, 256);
    Linear linear3(256, 128);
    Linear linear4(128, 64);
    Linear linear5(64, 10);

    try {
        /* Create Dataset and DataLoader */
        auto options = torch::data::DataLoaderOptions().batch_size(batch_size);
        
        auto dataset_train = MNISTDataset(train_path)
                           .map(torch::data::transforms::Normalize<>(0.0, 1.0))
                           .map(torch::data::transforms::Stack<>());

        auto data_loader_train = torch::data::make_data_loader(
            std::move(dataset_train), options);

        auto dataset_test = MNISTDataset(test_path)
                           .map(torch::data::transforms::Normalize<>(0.0, 1.0))
                           .map(torch::data::transforms::Stack<>());

        auto data_loader_test = torch::data::make_data_loader(
            std::move(dataset_test), options);

        int num_epochs = 1;
        int num_batches_train = std::distance(data_loader_train->begin(), data_loader_train->end());

        for (int epoch = 0; epoch < num_epochs; epoch++) {
            int batch_idx = 0;
            auto batch_it = data_loader_train->begin();

            while (batch_idx < num_batches_train) {
                std::vector<std::thread> workers;
                losses.clear();
                
                for (size_t t = 0; t < num_threads && batch_it != data_loader_train->end(); ++t, ++batch_it) {
                    auto& batch = *batch_it;

                    /* Flatten images to [batch_size, 784] */
                    auto flattened_images = batch.data.view({batch.data.size(0), -1}).to(torch::kCPU).contiguous();

                    /* Convert images to std::vector<float> */
                    std::vector<float> images_vec(flattened_images.numel());
                    std::memcpy(images_vec.data(), flattened_images.data_ptr<float>(), images_vec.size() * sizeof(float));

                    /* Convert labels to std::vector<float> */
                    std::vector<float> labels_vec(batch.target.numel());
                    std::memcpy(labels_vec.data(), batch.target.data_ptr<float>(), labels_vec.size() * sizeof(float));

                    size_t real_batch_size = batch.data.size(0);
                    Tensor x = Tensor(images_vec, {real_batch_size, 784}, false);
                    Tensor y = Tensor(labels_vec, {real_batch_size}, false);


                    workers.emplace_back(worker, x, y, std::ref(linear1), std::ref(linear2), std::ref(linear3), std::ref(linear4), std::ref(linear5), num_threads);
                }

                for (auto& t : workers) {
                    t.join();
                }

                float avg_loss = std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();

                std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                          << "Batch [" << (batch_idx + 1) << "/" << num_batches_train << "] "
                          << "Loss: " << avg_loss << std::endl;
                
                batch_idx += num_threads;
            }
        }
        std::cout << "============================<EVAL PHASE>============================" << std::endl;
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
            size_t real_batch_size = batch.data.size(0);
            Tensor x = Tensor(images_vec, {real_batch_size, 784}, false);

            Tensor y_pred = linear5.forward(
                                        linear4.forward(
                                                linear3.forward(
                                                        linear2.forward(
                                                                linear1.forward(x).relu()
                                                        ).relu()
                                                ).relu()
                                        )
                                );
            size_t pred_size = y_pred.data().size();
            std::vector<float> predictions(pred_size);
            std::memcpy(predictions.data(), y_pred.data().data(), pred_size * sizeof(float));

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
        std::cerr << "Training failed: " << e.what() << std::endl;
    }
}

int main() {
    float time = 0.0f;
    std::vector<size_t> num_threads = {4};
    std::vector<size_t > batch_sizes = {32};
    
    for (auto b: batch_sizes){
    for (auto t: num_threads) {
        time += ExecutionTimer::measure("train", [t, b]() {      train(t, b);    });
        std::cout << "=========================================================" << std::endl;
        std::cout << "Average time for training with " << t << " threads and batch size " << b << ": " << time << " ms" << std::endl;
        time = 0.0f;
    }
    }
}