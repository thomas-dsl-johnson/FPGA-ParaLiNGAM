#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <vector>
#include <iostream>

// Matrix struct
struct Matrix {
    std::vector<float> data;
    size_t rows;
    size_t cols;
};

// Kernel 1: standardize_data
void standardize_data(sycl::queue& q, sycl::buffer<float, 2>& buffer_x) {
    const auto num_rows = buffer_x.get_range()[0];
    const auto num_cols = buffer_x.get_range()[1];
    q.submit([&](sycl::handler& h) {
        auto accessor_x = buffer_x.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::range<1>(num_cols), [=](sycl::id<1> col_idx) {
            float sum = 0.0f;
            float sum_sq = 0.0f;
            for (size_t row = 0; row < num_rows; ++row) {
                float val = accessor_x[row][col_idx];
                sum += val;
                sum_sq += val * val;
            }
            float mean = sum / num_rows;
            float std_dev = sycl::sqrt(sum_sq / num_rows - mean * mean);
            if (std_dev > 1e-9f) {
                for (size_t row = 0; row < num_rows; ++row) {
                    accessor_x[row][col_idx] = (accessor_x[row][col_idx] - mean) / std_dev;
                }
            }
        });
    }).wait();
}

// Kernel 2: calculate_covariance
void calculate_covariance(sycl::queue& q, sycl::buffer<float, 2>& buffer_x, sycl::buffer<float, 2>& buffer_cov) {
    const auto num_rows = buffer_x.get_range()[0];
    const auto num_cols = buffer_x.get_range()[1];

    q.submit([&](sycl::handler& h) {
        auto accessor_x = buffer_x.get_access<sycl::access::mode::read>(h);
        auto accessor_cov = buffer_cov.get_access<sycl::access::mode::write>(h);
        
        h.parallel_for(sycl::range<2>(num_cols, num_cols), [=](sycl::id<2> idx) {
            size_t i = idx[0];
            size_t j = idx[1];

            if (i > j) return;

            float cov_sum = 0.0f;
            for (size_t k = 0; k < num_rows; ++k) {
                cov_sum += accessor_x[k][i] * accessor_x[k][j];
            }
            
            float covariance = cov_sum / (num_rows - 1);
            accessor_cov[i][j] = covariance;
            if (i != j) {
                accessor_cov[j][i] = covariance;
            }
        });
    }).wait();
}

// Main function to drive the kernels
int main() {
    Matrix m;
    m.rows = 100;
    m.cols = 5;
    m.data.resize(m.rows * m.cols);
    for (size_t i = 0; i < m.data.size(); ++i) {
        m.data[i] = static_cast<float>(i % 20);
    }
    
    #if defined(FPGA_HARDWARE)
        sycl::queue q(sycl::ext::intel::fpga_selector_v);
    #else
        sycl::queue q(sycl::ext::intel::fpga_emulator_selector_v);
    #endif

    sycl::buffer<float, 2> buffer_x(m.data.data(), sycl::range<2>(m.rows, m.cols));
    standardize_data(q, buffer_x);

    std::vector<float> cov_data(m.cols * m.cols);
    sycl::buffer<float, 2> buffer_cov(cov_data.data(), sycl::range<2>(m.cols, m.cols));
    calculate_covariance(q, buffer_x, buffer_cov);

    return 0;
}
