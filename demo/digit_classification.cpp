#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <mutex>
#include <numeric>
#include <chrono>
#include <unordered_map>

#include "../grad/cppgrad.h"
#include "../grad/modules.h"
#include "../performance_evals/timer.h"
#include "./data/mnist_dataset.h"


std::mutex grad_mutex;
std::vector<float> losses;

void worker(Tensor x, Tensor y, Linear &linear1, Linear &linear2, Linear &linear3, 
            Linear &linear4, Linear &linear5, size_t num_threads) {
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

    std::lock_guard<std::mutex> lock(grad_mutex);
    losses.push_back(loss.data()[0]);

    /* Update weights */
    for (auto layer : {&linear1, &linear2, &linear3, &linear4, &linear5}) {
        auto& data_weight = layer->weight.ptr->data;
        auto& grad_weight = layer->weight.grad().ptr->data;

        for (size_t i = 0; i < grad_weight.size(); i++) {
            data_weight[i] -= 0.1 * grad_weight[i];
        }

        auto& data_bias = layer->bias.ptr->data;
        auto& grad_bias = layer->bias.grad().ptr->data;

        for (size_t i = 0; i < grad_bias.size(); i++) {
            data_bias[i] -= 0.1 * grad_bias[i];
        }

        layer->weight.zero_grad();
        layer->bias.zero_grad();
        }   

}

template <typename DataLoader>
void train(const size_t num_threads, const size_t batch_size,
           Linear &linear1, Linear &linear2, Linear &linear3, Linear &linear4, Linear &linear5,
           DataLoader& data_loader_train) {
    try {
        size_t print = 0;
        for (auto layer : {&linear1, &linear2, &linear3, &linear4, &linear5}) {
            layer->train();
        }
        
        int num_epochs = 1;
            int num_batches_train = std::distance(data_loader_train.begin(), data_loader_train.end());

        for (int epoch = 0; epoch < num_epochs; epoch++) {
            int batch_idx = 0;
            auto batch_it = data_loader_train.begin();

            while (batch_idx < num_batches_train) {
                std::vector<std::thread> workers;
                losses.clear();
                
                for (size_t t = 0; t < num_threads && batch_it != data_loader_train.end(); ++t, ++batch_it) {
                    auto& batch = *batch_it;

                    /* Flatten images to [batch_size, 784] */
                    auto flattened_images = batch.data.view({batch.data.size(0), -1}).to(torch::kCPU).contiguous();

                    /* Convert images to std::vector<float> */
                    std::vector<float> images_vec(flattened_images.numel());
                    std::memcpy(images_vec.data(), flattened_images.template data_ptr<float>(), images_vec.size() * sizeof(float));

                    /* Convert labels to std::vector<float> */
                    std::vector<float> labels_vec(batch.target.numel());
                    std::memcpy(labels_vec.data(), batch.target.template data_ptr<float>(), labels_vec.size() * sizeof(float));

                    size_t real_batch_size = batch.data.size(0);
                    Tensor x = Tensor(images_vec, {real_batch_size, 784}, false);
                    Tensor y = Tensor(labels_vec, {real_batch_size}, false);

                    workers.emplace_back(worker, x, y, std::ref(linear1), std::ref(linear2), std::ref(linear3), std::ref(linear4), std::ref(linear5), num_threads);
                }

                for (auto& t : workers) {
                    t.join();
                }

                float avg_loss = std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();
                if (batch_idx >= print) {
                    std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                        << "Batch [" << (batch_idx + 1) << "/" << num_batches_train << "] "
                        << "Loss: " << avg_loss << std::endl;
                    print += 500;
                }
                
                batch_idx += num_threads;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Training failed: " << e.what() << std::endl;
    }
}

std::mutex stats_mutex;
int total_correct = 0;
int total_samples = 0;
std::unordered_map<int, int> correct_per_class;
std::unordered_map<int, int> total_per_class;

void worker_test(Tensor x, std::vector<float> labels_vec, Linear &linear1, Linear &linear2, Linear &linear3, 
                 Linear &linear4, Linear &linear5, size_t num_threads) {

    int local_total_correct = 0;
    int local_total_samples = 0;
    std::unordered_map<int, int> local_correct_per_class;
    std::unordered_map<int, int> local_total_per_class;
 
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
            local_total_correct++;
            local_correct_per_class[true_class]++;
        }

        local_total_per_class[true_class]++;
        local_total_samples++;
    }

    std::lock_guard<std::mutex> lock(stats_mutex);
    total_correct += local_total_correct;
    total_samples += local_total_samples;

    for (size_t c = 0; c < 10; c++) {
        correct_per_class[c] += local_correct_per_class[c];
        total_per_class[c] += local_total_per_class[c];
    }

}

template <typename DataLoader>
void test(const size_t num_threads, const size_t batch_size,
           Linear &linear1, Linear &linear2, Linear &linear3, Linear &linear4, Linear &linear5,
           DataLoader& data_loader_test) {
    try {

        for (auto layer : {&linear1, &linear2, &linear3, &linear4, &linear5}) {
            layer->eval();
        }
        
        int num_epochs = 1;
            int num_batches_train = std::distance(data_loader_test.begin(), data_loader_test.end());

        for (int epoch = 0; epoch < num_epochs; epoch++) {
            int batch_idx = 0;
            auto batch_it = data_loader_test.begin();

            while (batch_idx < num_batches_train) {
                std::vector<std::thread> workers;
                losses.clear();
                
                for (size_t t = 0; t < num_threads && batch_it != data_loader_test.end(); ++t, ++batch_it) {
                    auto& batch = *batch_it;

                    /* Flatten images to [batch_size, 784] */
                    auto flattened_images = batch.data.view({batch.data.size(0), -1}).to(torch::kCPU).contiguous();

                    /* Convert images to std::vector<float> */
                    std::vector<float> images_vec(flattened_images.numel());
                    std::memcpy(images_vec.data(), flattened_images.template data_ptr<float>(), images_vec.size() * sizeof(float));

                    /* Convert labels to std::vector<float> */
                    std::vector<float> labels_vec(batch.target.numel());
                    std::memcpy(labels_vec.data(), batch.target.template data_ptr<float>(), labels_vec.size() * sizeof(float));

                    size_t real_batch_size = batch.data.size(0);
                    Tensor x = Tensor(images_vec, {real_batch_size, 784}, false);

                    workers.emplace_back(worker_test, x, labels_vec, std::ref(linear1), std::ref(linear2), std::ref(linear3), std::ref(linear4), std::ref(linear5), num_threads);
                }

                for (auto& t : workers) {
                    t.join();
                }
                
                batch_idx += num_threads;
            }
        }

        /* Compute accuracies */
        float overall_accuracy = 100.0f * total_correct / total_samples;
        std::cout << "Overall Accuracy: " << overall_accuracy << "%" << std::endl;

        for (const auto& [class_label, correct] : correct_per_class) {
            float class_accuracy = 100.0f * correct / total_per_class[class_label];
            std::cout << "Accuracy for class " << class_label << ": " << class_accuracy << "%" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Testing failed: " << e.what() << std::endl;
    }
}

int main() {

    const std::string train_path = "../../demo/data/mnist_train.csv";
    const std::string test_path = "../../demo/data/mnist_test.csv";

    float train_time = 0.0f;
    float test_time = 0.0f;
    size_t num_threads = 4;
    size_t batch_size = 32;
    
    /* Define the model */
    Linear linear1(784, 512);
    Linear linear2(512, 256);
    Linear linear3(256, 128);
    Linear linear4(128, 64);
    Linear linear5(64, 10);

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

    train_time += ExecutionTimer::measure("train", [&] {    train(num_threads, batch_size, linear1, linear2, linear3, linear4, linear5, *data_loader_train);    });
    test_time += ExecutionTimer::measure("test", [&]() {    test(num_threads, batch_size, linear1, linear2, linear3, linear4, linear5, *data_loader_train);    });
    std::cout << "=========================================================" << std::endl;
    std::cout << "Total train time: " << train_time << " ms" << std::endl;
    std::cout << "Total test time: " << test_time << " ms" << std::endl;
    std::cout << "Total time with " << num_threads << " threads and batch size " << batch_size << ": " << train_time + test_time << " ms" << std::endl;
}