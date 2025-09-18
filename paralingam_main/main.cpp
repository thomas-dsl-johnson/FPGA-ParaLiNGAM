#include "ParaLingam.hpp"
#include <iostream>
#include <random>
#include <chrono>

// Generates the same sample data as the Python example.
Matrix get_matrix() {
    constexpr size_t n_samples = 1000;
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
    m.cols = 6;
    m.data.resize(n_samples * 6);

    // Store data in column-major format
    for (size_t i = 0; i < n_samples; ++i) {
        m.data[i + 0 * n_samples] = x0[i];
        m.data[i + 1 * n_samples] = x1[i];
        m.data[i + 2 * n_samples] = x2[i];
        m.data[i + 3 * n_samples] = x3[i];
        m.data[i + 4 * n_samples] = x4[i];
        m.data[i + 5 * n_samples] = x5[i];
    }
    return m;
}

int main() {
    std::cout << "Running Parallelised ParaLiNGAM Algorithm in DPC++..." << std::endl;

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