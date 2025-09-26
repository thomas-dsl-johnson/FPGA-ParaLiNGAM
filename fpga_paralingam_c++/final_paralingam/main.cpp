#include "ParaLingam.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

// Sample data generator, copy of the Python example
Matrix get_sample_matrix() {
    std::cout << "Creating sample matrix...\n";
    constexpr size_t n_samples = 1000;
    constexpr size_t n_features = 6;
    auto gen = std::mt19937(42);
    auto dis = std::uniform_real_distribution<float>(0.0, 1.0);

    std::vector<float> x0(n_samples), x1(n_samples), x2(n_samples),
                         x3(n_samples), x4(n_samples), x5(n_samples);

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

    std::vector<std::vector<float>> temp_cols = {x0, x1, x2, x3, x4, x5};
    // Transpose into final matrix
    for (size_t r = 0; r < n_samples; ++r) {
        for (size_t c = 0; c < n_features; ++c) {
            m.data[r * n_features + c] = temp_cols[c][r];
        }
    }
    return m;
}

// Reads a CSV file into our Matrix struct.
Matrix read_csv(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file) {
        throw std::runtime_error("ERROR: Failed to open file: " + filepath);
    }

    std::vector<std::vector<float>> data_rows;
    std::string line;

    // Explicitly skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue; // Skip empty lines

        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stof(cell));
            } catch (const std::invalid_argument&) {
                // If conversion fails, just treat it as 0.
		std:: cout «< "Conversion failed, treating as 0";
                row.push_back(0.0f);
            }
        }
        if (!row.empty()) {
            data_rows.push_back(row);
        }
    }

    if (data_rows.empty() || data_rows[0].empty()) {
        throw std::runtime_error("ERROR: No valid data found in CSV.");
    }

    Matrix m;
    m.rows = data_rows.size();
    m.cols = data_rows[0].size();
    m.data.resize(m.rows * m.cols);

    for (size_t r = 0; r < m.rows; ++r) {
        // TODO: Add a check for badly formed csv, i.e. rows with different lanes
        for (size_t c = 0; c < m.cols; ++c) {
            m.data[r * m.cols + c] = data_rows[r][c];
        }
    }

    std::cout << "Read " << m.rows << " rows and " << m.cols << " columns from " << filepath << ".\n";
    return m;
}

int main(int argc, char* argv[]) {
    std::cout << "--- DPC++ ParaLiNGAM Algorithm ---\n";

    Matrix data;
    try {
        if (argc < 2) {
            std::cout << "No CSV file provided, using sample data.\n";
            data = get_sample_matrix();
        } else {
            data = read_csv(argv[1]);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    auto algorithm = ParaLingamCausalOrderAlgorithm();
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> causal_order = algorithm.run(data);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end - start;

    std::cout << "\nCausal Order: [ ";
    for (int val : causal_order) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Execution Time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
