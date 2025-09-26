#include "ParaLingam.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Kernel for standardising data: mean=0, std=1 for each column
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
            // Calculate sample standard deviation (N-1 denominator) to match Python's ddof=1.
            // The unbiased estimator for variance is (sum_sq - n*mean^2) / (n-1)

            float std_dev = 0.0f;
            if (num_rows > 1) {
                float variance = sycl::fabs(sum_sq - num_rows * mean * mean) / (num_rows - 1);
                std_dev = sycl::sqrt(variance);
            }

            if (std_dev > 1e-9f) {
                for (size_t row = 0; row < num_rows; ++row) {
                    accessor_x[row][col_idx] = (accessor_x[row][col_idx] - mean) / std_dev;
                }
            }
        });
    }).wait();
}

// Kernel for calculating the covariance matrix of standardised data.
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
            
            // Calculate covariance (N denominator) to match Pythons bias=True
            float covariance = cov_sum / num_rows;
            accessor_cov[i][j] = covariance;
            if (i != j) {
                accessor_cov[j][i] = covariance;
            }
        });
    }).wait();
}

// KERNEL: Efficiently updates the covariance matrix based on the paper's Algorithm 8.
void ParaLingamCausalOrderAlgorithm::update_covariance(sycl::queue& q, sycl::buffer<float, 2>& current_cov_buf, sycl::buffer<float, 2>& next_cov_buf, int root_idx) {
    const auto n_remaining = next_cov_buf.get_range()[0];
    
    q.submit([&](sycl::handler& h) {
        auto current_cov = current_cov_buf.get_access<sycl::access::mode::read>(h);
        auto next_cov = next_cov_buf.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<2>(n_remaining, n_remaining), [=](sycl::id<2> idx) {
            size_t i_new = idx[0];
            size_t j_new = idx[1];

            if (i_new > j_new) return;

            // Map new indices to old indices
            size_t i_old = (i_new < root_idx) ? i_new : i_new + 1;
            size_t j_old = (j_new < root_idx) ? j_new : j_new + 1;

            float cov_ij = current_cov[i_old][j_old];
            float cov_ir = current_cov[i_old][root_idx];
            float cov_jr = current_cov[j_old][root_idx];

            float var_r_i = sycl::fabs(1.0f - cov_ir * cov_ir);
            float var_r_j = sycl::fabs(1.0f - cov_jr * cov_jr);
            
            float new_cov_ij = 0.0f;
            if (var_r_i > 1e-9f && var_r_j > 1e-9f) {
                new_cov_ij = (cov_ij - cov_ir * cov_jr) / (sycl::sqrt(var_r_i) * sycl::sqrt(var_r_j));
            }

            next_cov[i_new][j_new] = new_cov_ij;
            if (i_new != j_new) {
                next_cov[j_new][i_new] = new_cov_ij;
            } else {
                next_cov[i_new][j_new] = 1.0f; // Diagonal of standardised cov matrix is 1
            }
        });
    }).wait();
}


// Parallel root finding.
int ParaLingamCausalOrderAlgorithm::para_find_root(sycl::queue& q, sycl::buffer<float, 2>& buffer_x, sycl::buffer<float, 2>& buffer_cov) {
    const auto num_rows = buffer_x.get_range()[0];
    const auto num_cols = buffer_x.get_range()[1];

    if (num_cols <= 1) return 0;

    sycl::buffer<float, 2> buffer_partial_scores(sycl::range<2>(num_cols, num_cols));

    q.submit([&](sycl::handler& h) {
        auto accessor_x = buffer_x.get_access<sycl::access::mode::read>(h);
        auto accessor_cov = buffer_cov.get_access<sycl::access::mode::read>(h);
        auto accessor_partial = buffer_partial_scores.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<2>(num_cols, num_cols), [=](sycl::id<2> idx) {
            size_t i = idx[0];
            size_t j = idx[1];

            accessor_partial[i][j] = 0.0f;
            if (i >= j) return;

            // Entropy is calculated on-the-fly for a given residual type (r_i_j or r_j_i)
            auto calculate_entropy = [=](bool is_rij) -> float {
                constexpr float k1 = 79.047f;
                constexpr float k2 = 7.4129f;
                constexpr float gamma = 0.37457f;
                
                float cov_ij = accessor_cov[i][j];
                
                float sum_log_cosh = 0.0f;
                float sum_u_exp = 0.0f;
                float sum = 0.0f;
                float sum_sq = 0.0f;

                // First pass to calculate mean and std_dev of the residual
                for(size_t k = 0; k < num_rows; k++) {
                   float residual = is_rij ? (accessor_x[k][i] - cov_ij * accessor_x[k][j])
                                           : (accessor_x[k][j] - cov_ij * accessor_x[k][i]);
                   sum += residual;
                   sum_sq += residual * residual;
                }
                float mean = sum / num_rows;
                // This uses population std dev (N denominator), matching Python's inner np.std()
                float std_dev = sycl::sqrt(sycl::fabs(sum_sq / num_rows - mean * mean));

                if (std_dev < 1e-9f) return 0.0f;

                // Second pass to calculate entropy terms with the standardised residual
                for(size_t k = 0; k < num_rows; k++) {
                   float residual = is_rij ? (accessor_x[k][i] - cov_ij * accessor_x[k][j])
                                           : (accessor_x[k][j] - cov_ij * accessor_x[k][i]);
                   float u = (residual - mean) / std_dev;
                   sum_log_cosh += sycl::log(sycl::cosh(u));
                   sum_u_exp += u * sycl::exp(-0.5f * u * u);
                }
                float mean_log_cosh = sum_log_cosh / num_rows;
                float mean_u_exp = sum_u_exp / num_rows;
                
                float term1 = (mean_log_cosh - gamma);
                float term2 = mean_u_exp;
                
                return (1.0f + sycl::log(2.0f * M_PI))/2.0f - k1 * term1 * term1 - k2 * term2 * term2;
            };

            float h_ri_j = calculate_entropy(true);
            float h_rj_i = calculate_entropy(false);
            float diff_mi = h_ri_j - h_rj_i;
            
            float score_contribution_i = sycl::min(0.0f, diff_mi) * sycl::min(0.0f, diff_mi);
            float score_contribution_j = sycl::min(0.0f, -diff_mi) * sycl::min(0.0f, -diff_mi);

            accessor_partial[i][j] = score_contribution_i;
            accessor_partial[j][i] = score_contribution_j;
        });
    }).wait();

    // Step 2: Reduce Kernel.
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

    // Create initial matrix and buffers
    Matrix current_matrix = matrix;
    sycl::buffer<float, 2> current_x_buf(current_matrix.data.data(), sycl::range<2>(n_samples, n_features));
    sycl::buffer<float, 2> current_cov_buf(sycl::range<2>(n_features, n_features));

    standardize_data(q, current_x_buf);
    calculate_covariance(q, current_x_buf, current_cov_buf);

    for (size_t k = 0; k < n_features - 1; ++k) {
        int root_idx_in_current = para_find_root(q, current_x_buf, current_cov_buf);
        
        int original_root_idx = U[root_idx_in_current];
        K.push_back(original_root_idx);
        U.erase(U.begin() + root_idx_in_current);

        size_t n_remaining = U.size();
        Matrix next_matrix;
        next_matrix.rows = n_samples;
        next_matrix.cols = n_remaining;
        next_matrix.data.resize(n_samples * n_remaining);
        
        sycl::buffer<float, 2> next_x_buf(next_matrix.data.data(), sycl::range<2>(n_samples, n_remaining));
        sycl::buffer<float, 2> next_cov_buf(sycl::range<2>(n_remaining, n_remaining));

        // Update data by regressing out the root
        q.submit([&](sycl::handler& h) {
            auto accessor_x = current_x_buf.get_access<sycl::access::mode::read>(h);
            auto accessor_cov = current_cov_buf.get_access<sycl::access::mode::read>(h);
            auto next_accessor = next_x_buf.get_access<sycl::access::mode::write>(h);

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

        // Update Covariance
        update_covariance(q, current_cov_buf, next_cov_buf, root_idx_in_current);
        
        // Move data and buffers for the next iteration
        current_matrix = std::move(next_matrix);
        current_x_buf = std::move(next_x_buf);
        current_cov_buf = std::move(next_cov_buf);
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
    
    #if defined(FPGA_EMULATOR)
        sycl::queue q(sycl::ext::intel::fpga_emulator_selector_v, exception_handler);
    #elif defined(FPGA_HARDWARE)
        sycl::queue q(sycl::ext::intel::fpga_selector_v, exception_handler);
    #elif defined(GPU)
        sycl::queue q(sycl::gpu_selector_v, exception_handler);
    #else
        sycl::queue q(sycl::cpu_selector_v, exception_handler);
    #endif

    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    
    return get_causal_order_using_paralingam(q, matrix);
}

std::string ParaLingamCausalOrderAlgorithm::to_string() const {
    return "ParaLingamAlgorithm";
}
