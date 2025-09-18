#include "ParaLingam.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>

// This header is now included for the FPGA selectors.
#include <sycl/ext/intel/fpga_extensions.hpp>

// DPC++ Kernel for standardizing data (mean=0, std=1 for each column).
void ParaLingamCausalOrderAlgorithm::standardize_data(sycl::queue& q, sycl::buffer<float, 2>& buffer_x) {
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

// DPC++ Kernel for calculating the covariance matrix of standardized data.
void ParaLingamCausalOrderAlgorithm::calculate_covariance(sycl::queue& q, sycl::buffer<float, 2>& buffer_x, sycl::buffer<float, 2>& buffer_cov) {
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
            
            float covariance = cov_sum / (num_rows -1);
            accessor_cov[i][j] = covariance;
            if (i != j) {
                accessor_cov[j][i] = covariance;
            }
        });
    }).wait();
}

// DPC++ implementation of the parallel root finding.
// DPC++ implementation of the parallel root finding (REVISED to avoid atomics).
// DPC++ implementation of the parallel root finding (REVISED for memory efficiency on FPGA).
// DPC++ implementation of the parallel root finding (REVISED to avoid atomics).
int ParaLingamCausalOrderAlgorithm::para_find_root(sycl::queue& q, sycl::buffer<float, 2>& buffer_x, sycl::buffer<float, 2>& buffer_cov) {
    const auto num_rows = buffer_x.get_range()[0];
    const auto num_cols = buffer_x.get_range()[1];

    if (num_cols <= 1) return 0;

    // Step 1: "Scatter" Kernel.
    // Calculate partial scores for each pair (i, j) and store them in a temporary
    // matrix. This avoids the need for atomic operations by giving each pair
    // its own unique memory location to write to.
    sycl::buffer<float, 2> buffer_partial_scores(sycl::range<2>(num_cols, num_cols));

    q.submit([&](sycl::handler& h) {
        auto accessor_x = buffer_x.get_access<sycl::access::mode::read>(h);
        auto accessor_cov = buffer_cov.get_access<sycl::access::mode::read>(h);
        auto accessor_partial = buffer_partial_scores.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<2>(num_cols, num_cols), [=](sycl::id<2> idx) {
            size_t i = idx[0];
            size_t j = idx[1];

            // Initialize the cell to 0 before proceeding.
            accessor_partial[i][j] = 0.0f;
            
            // We only need to compute for the upper triangle of the matrix (where j > i).
            if (i >= j) return;

            auto entropy = [=](const float* u_ptr) -> float {
                constexpr float k1 = 79.047f;
                constexpr float k2 = 7.4129f;
                constexpr float gamma = 0.37457f;

                float sum_log_cosh = 0.0f;
                float sum_u_exp = 0.0f;

                for(size_t k = 0; k < num_rows; k++) {
                   sum_log_cosh += sycl::log(sycl::cosh(u_ptr[k]));
                   sum_u_exp += u_ptr[k] * sycl::exp(-0.5f * u_ptr[k] * u_ptr[k]);
                }
                float mean_log_cosh = sum_log_cosh / num_rows;
                float mean_u_exp = sum_u_exp / num_rows;
                
                float term1 = (mean_log_cosh - gamma);
                float term2 = mean_u_exp;
                
                return (1.0f + sycl::log(2.0f * M_PI))/2.0f - k1 * term1 * term1 - k2 * term2 * term2;
            };

            float cov_ij = accessor_cov[i][j];
            float ri_j[2048], rj_i[2048]; 

            for (size_t k = 0; k < num_rows; k++) {
                ri_j[k] = accessor_x[k][i] - cov_ij * accessor_x[k][j];
                rj_i[k] = accessor_x[k][j] - cov_ij * accessor_x[k][i];
            }

            float h_ri_j = entropy(ri_j);
            float h_rj_i = entropy(rj_i);
            float diff_mi = h_ri_j - h_rj_i;
            
            float score_contribution_i = sycl::min(0.0f, diff_mi) * sycl::min(0.0f, diff_mi);
            float score_contribution_j = sycl::min(0.0f, -diff_mi) * sycl::min(0.0f, -diff_mi);

            // Write the partial scores to unique locations in the matrix.
            accessor_partial[i][j] = score_contribution_i;
            accessor_partial[j][i] = score_contribution_j;
        });
    }).wait();

    // Step 2: "Reduce" Kernel.
    // Sum the partial scores for each variable (i.e., each row of the partial matrix)
    // into the final scores vector.
    // CORRECT
    std::vector<float> host_scores(num_cols, 0.0f);
    sycl::buffer<float, 1> buffer_scores(host_scores.data(), sycl::range<1>(num_cols));

    q.submit([&](sycl::handler& h) {
        auto accessor_partial = buffer_partial_scores.get_access<sycl::access::mode::read>(h);
        auto accessor_scores = buffer_scores.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<1>(num_cols), [=](sycl::id<1> i) {
            float total_score = 0.0f;
            for (size_t j = 0; j < num_cols; ++j) {
                total_score += accessor_partial[i][j];
            }
            accessor_scores[i] = total_score;
        });
    }).wait();

    // Step 3: Find the minimum score on the host.
    auto final_scores = buffer_scores.get_host_access();
    int root_idx = std::distance(final_scores.begin(), std::min_element(final_scores.begin(), final_scores.end()));
    return root_idx;
}

std::vector<int> ParaLingamCausalOrderAlgorithm::get_causal_order_using_paralingam(sycl::queue& q, const Matrix& matrix) {
    size_t n_samples = matrix.rows;
    size_t n_features = matrix.cols;

    std::vector<int> U;
    for(int i=0; i<n_features; ++i) U.push_back(i);
    std::vector<int> K;

    Matrix current_matrix = matrix;

    for (size_t k = 0; k < n_features - 1; ++k) {
        sycl::buffer<float, 2> buffer_x(current_matrix.data.data(), sycl::range<2>(current_matrix.rows, current_matrix.cols));
        sycl::buffer<float, 2> buffer_cov(sycl::range<2>(current_matrix.cols, current_matrix.cols));

        standardize_data(q, buffer_x);
        calculate_covariance(q, buffer_x, buffer_cov);
        
        int root_idx_in_current = para_find_root(q, buffer_x, buffer_cov);
        
        int original_root_idx = U[root_idx_in_current];
        K.push_back(original_root_idx);
        U.erase(U.begin() + root_idx_in_current);

        size_t n_remaining = current_matrix.cols - 1;
        Matrix next_matrix;
        next_matrix.rows = n_samples;
        next_matrix.cols = n_remaining;
        next_matrix.data.resize(n_samples * n_remaining);

        {
            sycl::buffer<float, 2> buffer_next_x(next_matrix.data.data(), sycl::range<2>(n_samples, n_remaining));

            q.submit([&](sycl::handler& h) {
                auto accessor_x = buffer_x.get_access<sycl::access::mode::read>(h);
                auto accessor_cov = buffer_cov.get_access<sycl::access::mode::read>(h);
                auto next_accessor = buffer_next_x.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<1>(n_remaining), [=](sycl::id<1> i_rem) {
                    int i_curr = (i_rem[0] < root_idx_in_current) ? i_rem[0] : i_rem[0] + 1;
                    float cov_ir = accessor_cov[i_curr][root_idx_in_current];
                    float residual_std_dev = sycl::sqrt(sycl::fabs(1.0f - cov_ir * cov_ir));

                    for (size_t row = 0; row < n_samples; ++row) {
                        float residual = accessor_x[row][i_curr] - cov_ir * accessor_x[row][root_idx_in_current];
                        if (residual_std_dev > 1e-9f) {
                            next_accessor[row][i_rem] = residual / residual_std_dev;
                        } else {
                            next_accessor[row][i_rem] = 0.0f;
                        }
                    }
                });
            }).wait();
        } 
        current_matrix = std::move(next_matrix);
    }
    
    if (!U.empty()) {
        K.push_back(U[0]);
    }

    return K;
}

std::vector<int> ParaLingamCausalOrderAlgorithm::run(const Matrix& matrix) {
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch (sycl::exception const& e) {
                std::cerr << "Caught SYCL exception: " << e.what() << std::endl;
            }
        }
    };
    
    // Use preprocessor macros to select the device based on compiler flags.
    #if defined(FPGA_EMULATOR)
        sycl::queue q(sycl::ext::intel::fpga_emulator_selector_v, exception_handler);
    #elif defined(FPGA_HARDWARE) // This case is added for the report/hardware flow
        sycl::queue q(sycl::ext::intel::fpga_selector_v, exception_handler);
    #elif defined(GPU)
        sycl::queue q(sycl::gpu_selector_v, exception_handler);
    #else
        // Default to the CPU selector if no specific target is defined
        sycl::queue q(sycl::cpu_selector_v, exception_handler);
    #endif

    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    
    return get_causal_order_using_paralingam(q, matrix);
}

std::string ParaLingamCausalOrderAlgorithm::to_string() const {
    return "ParaLingamAlgorithm";
}