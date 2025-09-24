#include "ParaLingam.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

// Generates the same sample data as the Python example.
Matrix get_matrix() {
    std::cout << "Generating sample matrix..." << std::endl;
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

    std::vector<std::vector<float>> temp_data = {x0, x1, x2, x3, x4, x5};
    for (size_t r = 0; r < n_samples; ++r) {
        for (size_t c = 0; c < n_features; ++c) {
            m.data[r * n_features + c] = temp_data[c][r];
        }
    }
    return m;
}


// Reads a CSV file into the Matrix struct.
Matrix read_csv(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::vector<std::vector<float>> records;
    std::string line;
    
    // This correctly skips a header row if one exists.
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        
        // FIX: Added a column counter to skip the first column (index 0).
        int col_idx = 0;
        while (std::getline(ss, cell, ',')) {
            if (col_idx > 0) { // Only process columns after the first one.
                try {
                    row.push_back(std::stof(cell));
                } catch (const std::invalid_argument& e) {
                    // Python's `coerce` turns invalid values into NaN, which are then filled with 0.
                    // This mimics that behavior for robust parsing.
                    row.push_back(0.0f);
                }
            }
            col_idx++;
        }
        if (!row.empty()) {
            records.push_back(row);
        }
    }

    if (records.empty()) {
        throw std::runtime_error("No numeric data found in CSV file.");
    }

    Matrix m;
    m.rows = records.size();
    m.cols = records[0].size();
    m.data.resize(m.rows * m.cols);

    for (size_t r = 0; r < m.rows; ++r) {
        for (size_t c = 0; c < m.cols; ++c) {
            m.data[r * m.cols + c] = records[r][c];
        }
    }

    std::cout << "Successfully read " << m.rows << " rows and " << m.cols << " columns from " << filepath << std::endl;
    return m;
}

int main(int argc, char* argv[]) {
    std::cout << "Running FAITHFUL and OPTIMIZED ParaLiNGAM Algorithm in DPC++..." << std::endl;

    ParaLingamCausalOrderAlgorithm algorithm;
    Matrix data;

    try {
        if (argc < 2) {
            // If no CSV is provided, use the generated matrix.
            std::cout << "No CSV file provided. Falling back to sample data generator." << std::endl;
            data = get_matrix();
        } else {
            // If a CSV file is provided, read it.
            std::string filepath = argv[1];
            data = read_csv(filepath);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

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