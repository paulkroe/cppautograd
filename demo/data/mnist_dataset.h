#ifndef MNIST_DATASET_H
#define MNIST_DATASET_H
#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
    
/**
 * @brief A custom dataset class for loading MNIST data from a CSV file.
 *
 * This dataset reads an MNIST-formatted CSV file where each row consists of:
 * - A label (integer from 0 to 9) as the first column.
 * - 784 pixel values (flattened 28x28 image) normalized to the range [0, 1].
 *
 * The dataset stores images as 1x28x28 tensors and labels as float tensors.
 * It inherits from `torch::data::datasets::Dataset<MNISTDataset>` and implements
 * the `get()` and `size()` methods for integration with PyTorch's data loading utilities.
 *
 * @note This class expects a CSV file where each row has 785 comma-separated values
 *       (1 label + 784 pixel values). Any invalid row will trigger an exception.
 *
 * @param csv_path Path to the CSV file containing the dataset.
 */
class MNISTDataset : public torch::data::datasets::Dataset<MNISTDataset> {
    std::vector<torch::Tensor> images_;  ///< Vector storing image tensors (1x28x28).
    std::vector<torch::Tensor> labels_;  ///< Vector storing label tensors as floats.

public:
    /**
     * @brief Constructs an MNIST dataset from a CSV file.
     *
     * Reads the MNIST dataset from the specified CSV file, normalizing pixel values
     * and storing images and labels as tensors.
     *
     * @param csv_path Path to the MNIST CSV file.
     * @throws std::runtime_error If the file cannot be opened or contains invalid rows.
     */
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

    /**
     * @brief Retrieves a single sample from the dataset.
     *
     * @param index The index of the sample to retrieve.
     * @return torch::data::Example<> A data example containing the image tensor and label tensor.
     */
    torch::data::Example<> get(size_t index) override {
        return {images_[index], labels_[index]};
    }

    /**
     * @brief Returns the total number of samples in the dataset.
     *
     * @return torch::optional<size_t> The size of the dataset.
     */
    torch::optional<size_t> size() const override {
        return images_.size();
    }
};

#endif // MNIST_DATASET_H