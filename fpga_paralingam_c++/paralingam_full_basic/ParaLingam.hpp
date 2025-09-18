#ifndef PARALINGAM_HPP
#define PARALINGAM_HPP

#include <sycl/sycl.hpp>
#include <vector>
#include <string>

// A simple structure to hold the matrix data and its dimensions.
struct Matrix {
    std::vector<float> data;
    size_t rows;
    size_t cols;
};

class ParaLingamCausalOrderAlgorithm {
public:
    // Runs the causal order algorithm.
    std::vector<int> run(const Matrix& matrix);
    std::string to_string() const;

private:
    // Main algorithm logic implemented using DPC++.
    std::vector<int> get_causal_order_using_paralingam(sycl::queue& q, const Matrix& matrix);

    // DPC++ helper functions for various stages of the algorithm.
    // FIX 2: Removed const from buffer parameters to match .cpp definitions.
    void standardize_data(sycl::queue& q, sycl::buffer<float, 2>& buffer_x);
    void calculate_covariance(sycl::queue& q, sycl::buffer<float, 2>& buffer_x, sycl::buffer<float, 2>& buffer_cov);
    int para_find_root(sycl::queue& q, sycl::buffer<float, 2>& buffer_x, sycl::buffer<float, 2>& buffer_cov);
};

#endif // PARALINGAM_HPP