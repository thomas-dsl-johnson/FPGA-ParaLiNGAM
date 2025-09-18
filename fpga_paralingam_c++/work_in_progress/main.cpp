#include "ParaLingam.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

// Reads a CSV file into the Matrix struct.
Matrix read_csv(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::vector<std::vector<float>> records;
    std::string line;

    // Read the file line by line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        // Split the line by commas
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stof(cell));
            } catch (const std::invalid_argument& e) {
                // Skips non-numeric cells (like headers)
            }
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

    // Populate the Matrix data in row-major format
    for (size_t r = 0; r < m.rows; ++r) {
        for (size_t c = 0; c < m.cols; ++c) {
            m.data[r * m.cols + c] = records[r][c];
        }
    }

    std::cout << "Successfully read " << m.rows << " rows and " << m.cols << " columns from " << filepath << std::endl;
    return m;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_csv_file>" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    std::cout << "Running FAITHFUL and OPTIMIZED ParaLiNGAM Algorithm in DPC++..." << std::endl;

    ParaLingamCausalOrderAlgorithm algorithm;
    Matrix data;

    try {
        data = read_csv(filepath);
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

