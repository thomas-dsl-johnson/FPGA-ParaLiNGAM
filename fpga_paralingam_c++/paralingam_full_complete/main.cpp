#include "ParaLingam.hpp"
#include <iostream>
#include <random>
#include <chrono>

// Generates the same sample data as the Python example.
Matrix get_matrix() {
    constexpr size_t n_samples = 1000;
    constexpr size_t n_features = 6;
    
    std::vector<float> x0(n_samples), x1(n_samples), x2(n_samples),
                         x3(n_samples), x4(n_samples), x5(n_samples);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (size_t i = 0; i < n_samples; ++i) x3[i] = dis(gen);
    for (size_t i = 0; i < n_samples; ++i) x0[i] = 3.0f * x3[i] + dis(gen);
    for (size_t i = 0; i < n_samples; ++i) x2[i] = 6.0f * x3[i] + dis(gen);
    for (size_t i = 0; i < n_samples; ++i) x1[i] = 3.0f * x0[i] + 2.0f * x2[i] + dis(gen);
    for (size_t i = 0; i < n_samples; ++i) x5[i] = 4.0f * x0[i] + dis(gen);
    for (size_t i = 0; i < n_samples; ++i) x4[i] = 8.0f * x0[i] - 1.0f * x2[i] + dis(gen);

    Matrix m;
    m.rows = n_samples;
    m.cols = n_features;
    m.data.resize(n_samples * n_features);

    // **FIXED**: Store data in row-major format for efficient SYCL access.
    std::vector<std::vector<float>> temp_data = {x0, x1, x2, x3, x4, x5};
    for (size_t r = 0; r < n_samples; ++r) {
        for (size_t c = 0; c < n_features; ++c) {
            m.data[r * n_features + c] = temp_data[c][r];
        }
    }
    return m;
}

int main() {
    std::cout << "Running FAITHFUL and OPTIMIZED ParaLiNGAM Algorithm in DPC++..." << std::endl;

    ParaLingamCausalOrderAlgorithm algorithm;
    Matrix data = get_matrix();

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> causal_order = algorithm.run(data);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << "Causal Order: [ ";
    for (int val : causal_order) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Execution Time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
