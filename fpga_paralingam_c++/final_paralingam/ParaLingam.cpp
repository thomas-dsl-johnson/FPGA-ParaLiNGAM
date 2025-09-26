#include "ParaLingam.hpp"
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <algorithm>

std::vector<int> ParaLingamCausalOrderAlgorithm::run(const Matrix& data) {
    if (data.data.empty()) {
        throw std::invalid_argument("The input data matrix is empty");
    }
    const auto rows = data.rows;
    const auto cols = data.cols;

    // 1. We standardise the data 
    // Aim: set the mean to 0, set the variance to 1
    std::vector<float> standardised_data(data.data);
    std::vector<float> means(cols, 0.0f);
    std::vector<float> std_devs(cols, 0.0f);
    
    // Calculate the mean of each column
    for (size_t c = 0; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r) {
            means[c] += standardised_data[r * cols + c];
        }
        means[c] /= rows;
    }

    // Calculate the std devs of each column
    for (size_t c = 0; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r) {
            float diff = standardised_data[r * cols + c] - means[c];
            std_devs[c] += diff * diff;
        }
        std_devs[c] = std::sqrt(std_devs[c] / rows);
    }

    // Standardise the data in each column
    // Using the formula: (value - mean) / std_dev
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            if (std_devs[c] > 1e-6) { 
            // Avoid division by zero
                standardised_data[r * cols + c] = (standardised_data[r * cols + c] - means[c]) / std_devs[c];
            } else {
            // Column has no standard deviation, all values zeroed
                standardised_data[r * cols + c] = 0.0f;
            }
        }
    }

    // 2. Find the causal order
    // Causal Order
    std::vector<int> causal_order;
    causal_order.reserve(cols);
    // Unobserved Indices [0,1,2,...,n-1]
    std::vector<int> unobserved_indices(cols);
    std::iota(unobserved_indices.begin(), unobserved_indices.end(), 0);
    
    for (int i = 0; i < cols; ++i) {
        std::vector<float> values;
        values.reserve(unobserved_indices.size());

        for (int current_var : unobserved_indices) {
            std::vector<float> residuals = standardised_data; // Start with original data
            
            // Regress out the effect of already found variables
            for (int found_var : causal_order) {
                float dot_product = 0.0f;
                float found_var_norm_sq = 0.0f;
                for (size_t r = 0; r < rows; ++r) {
                    dot_product += residuals[r * cols + current_var] * residuals[r * cols + found_var];
                    found_var_norm_sq += residuals[r * cols + found_var] * residuals[r * cols + found_var];
                }
                
                if (found_var_norm_sq > 1e-6) {
                    float beta = dot_product / found_var_norm_sq;
                    for (size_t r = 0; r < rows; ++r) {
                        residuals[r * cols + current_var] -= beta * residuals[r * cols + found_var];
                    }
                }
            }
            
            // Calculate a measure of non-Gaussianity for the residuals
            float mean_sq = 0.0f;
            float mean_fourth = 0.0f;
            for (size_t r = 0; r < rows; ++r) {
                float res = residuals[r * cols + current_var];
                mean_sq += res * res;
                mean_fourth += res * res * res * res;
            }
            mean_sq /= rows;
            mean_fourth /= rows;
            
            // This ratio is a metric for the shape of the distribution
            values.push_back((mean_sq > 1e-6) ? (mean_fourth / (mean_sq * mean_sq)) : 0.0f);
        }

        // Find the variable with the minimum value for the shape metric
        auto min_it = std::min_element(values.begin(), values.end());
        int min_idx = std::distance(values.begin(), min_it);
        
        int next_causal_var = unobserved_indices[min_idx];
        causal_order.push_back(next_causal_var);
        
        // Remove the found variable from the unobserved list
        unobserved_indices.erase(unobserved_indices.begin() + min_idx);
    }
    
    return causal_order;
}
