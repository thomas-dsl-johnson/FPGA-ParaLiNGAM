#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <vector>
#include <iostream>

// A simple structure to hold the matrix data and its dimensions.
struct Matrix {
    std::vector<float> data;
    size_t rows;
    size_t cols;
};

// DPC++ Kernel for standardizing data (mean=0, std=1 for each column).
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

// Main function to drive the kernel
int main() {
    // Create some sample data
    Matrix m;
    m.rows = 100;
    m.cols = 5;
    m.data.resize(m.rows * m.cols);
    for (size_t i = 0; i < m.data.size(); ++i) {
        m.data[i] = static_cast<float>(i);
    }

    // Exception handler
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch (sycl::exception const& e) {
                std::cerr << "Caught SYCL exception: " << e.what() << std::endl;
            }
        }
    };

    #if defined(FPGA_EMULATOR)
        sycl::queue q(sycl::ext::intel::fpga_emulator_selector_v, exception_handler);
    #elif defined(FPGA_HARDWARE)
        sycl::queue q(sycl::ext::intel::fpga_selector_v, exception_handler);
    #else
        sycl::queue q(sycl::cpu_selector_v, exception_handler);
    #endif

    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Create a SYCL buffer from the matrix data
    sycl::buffer<float, 2> buffer_x(m.data.data(), sycl::range<2>(m.rows, m.cols));

    std::cout << "Running standardize_data kernel..." << std::endl;
    standardize_data(q, buffer_x);
    std::cout << "Kernel execution finished." << std::endl;

    return 0;
}
