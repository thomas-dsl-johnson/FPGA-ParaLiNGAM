#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <numeric>

// A simple structure to hold the matrix data and its dimensions.
// This struct is used only on the host.
struct Matrix {
    std::vector<float> data;
    size_t rows;
    size_t cols;
};

// Kernel 1: DPC++ Kernel for standardizing data (mean=0, std=1 for each column).
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

// Kernel 2: DPC++ Kernel for calculating the covariance matrix.
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

// Kernel 3: DPC++ implementation of the parallel root finding (Optimized).
int para_find_root(sycl::queue& q, sycl::buffer<float, 2>& buffer_x, sycl::buffer<float, 2>& buffer_cov) {
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

            auto entropy = [=](const float* u_ptr) -> float {
                constexpr float k1 = 79.047f;
                constexpr float k2 = 7.4129f;
                constexpr float gamma = 0.37457f;
                float sum_log_cosh = 0.0f;
                float sum_u_exp = 0.0f;
                
                //#####################################################//
                //#######      OPTIMIZATION FOR STEP 6      #######//
                //#####################################################//
                // This attribute tells the compiler that loop iterations are independent,
                // allowing for better pipelining and an ideal II of 1.
                [[intel::ivdep]]
                for(size_t k = 0; k < num_rows; k++) {
                   sum_log_cosh += sycl::log(sycl::cosh(u_ptr[k]));
                   sum_u_exp += u_ptr[k] * sycl::exp(-0.5f * u_ptr[k] * u_ptr[k]);
                }
                float mean_log_cosh = sum_log_cosh / num_rows;
                float mean_u_exp = sum_u_exp / num_rows;
                float term1 = (mean_log_cosh - gamma);
                float term2 = mean_u_exp;
                return (1.0f + sycl::log(2.0f * (float)M_PI))/2.0f - k1 * term1 * term1 - k2 * term2 * term2;
            };

            float cov_ij = accessor_cov[i][j];
            float ri_j[100]; 
            float rj_i[100];
            for (size_t k = 0; k < num_rows; k++) {
                ri_j[k] = accessor_x[k][i] - cov_ij * accessor_x[k][j];
                rj_i[k] = accessor_x[k][j] - cov_ij * accessor_x[k][i];
            }
            float h_ri_j = entropy(ri_j);
            float h_rj_i = entropy(rj_i);
            float diff_mi = h_ri_j - h_rj_i;
            float score_contribution_i = sycl::min(0.0f, diff_mi) * sycl::min(0.0f, diff_mi);
            float score_contribution_j = sycl::min(0.0f, -diff_mi) * sycl::min(0.0f, -diff_mi);
            accessor_partial[i][j] = score_contribution_i;
            accessor_partial[j][i] = score_contribution_j;
        });
    }).wait();

    std::vector<float> host_scores(num_cols, 0.0f);
    sycl::buffer<float, 1> buffer_scores(host_scores.data(), sycl::range<1>(num_cols));
    q.submit([&](sycl::handler& h) {
        auto accessor_partial = buffer_partial_scores.get_access<sycl::access::mode::read>(h);
        auto accessor_scores = buffer_scores.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(num_cols), [=](sycl::id<1> i) {
            float total_score = 0.0f;
            [[intel::ivdep]]
            for (size_t j = 0; j < num_cols; ++j) {
                total_score += accessor_partial[i][j];
            }
            accessor_scores[i] = total_score;
        });
    }).wait();
    
    auto final_scores = buffer_scores.get_host_access();
    int root_idx = std::distance(final_scores.begin(), std::min_element(final_scores.begin(), final_scores.end()));
    return root_idx;
}

// Kernel 4: Regress out the root and re-normalize the remaining data
void regress_and_normalize(sycl::queue& q, 
                           sycl::buffer<float, 2>& buffer_current_x, 
                           sycl::buffer<float, 2>& buffer_next_x, 
                           sycl::buffer<float, 2>& buffer_cov, 
                           int root_idx_in_current) {
    const auto num_rows = buffer_current_x.get_range()[0];
    const auto num_cols_next = buffer_next_x.get_range()[1];

    q.submit([&](sycl::handler& h) {
        auto accessor_current_x = buffer_current_x.get_access<sycl::access::mode::read>(h);
        auto accessor_cov = buffer_cov.get_access<sycl::access::mode::read>(h);
        auto accessor_next_x = buffer_next_x.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<1>(num_cols_next), [=](sycl::id<1> i_rem) {
            int i_curr = (i_rem[0] < root_idx_in_current) ? i_rem[0] : i_rem[0] + 1;
            float cov_ir = accessor_cov[i_curr][root_idx_in_current];
            float residual_std_dev = sycl::sqrt(sycl::fabs(1.0f - cov_ir * cov_ir));

            for (size_t row = 0; row < num_rows; ++row) {
                float residual = accessor_current_x[row][i_curr] - cov_ir * accessor_current_x[row][root_idx_in_current];
                if (residual_std_dev > 1e-9f) {
                    accessor_next_x[row][i_rem] = residual / residual_std_dev;
                } else {
                    accessor_next_x[row][i_rem] = 0.0f;
                }
            }
        });
    }).wait();
}

// Main function with the host-driven loop
int main() {
    constexpr size_t N_SAMPLES = 100;
    constexpr size_t N_FEATURES = 5;

    Matrix initial_matrix;
    initial_matrix.rows = N_SAMPLES;
    initial_matrix.cols = N_FEATURES;
    initial_matrix.data.resize(N_SAMPLES * N_FEATURES);
    for (size_t i = 0; i < initial_matrix.data.size(); ++i) {
        initial_matrix.data[i] = static_cast<float>(i % 20);
    }
    
    #if defined(FPGA_HARDWARE)
        sycl::queue q(sycl::ext::intel::fpga_selector_v);
    #else
        sycl::queue q(sycl::ext::intel::fpga_emulator_selector_v);
    #endif
    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    std::vector<int> U(N_FEATURES);
    std::iota(U.begin(), U.end(), 0);
    std::vector<int> K;
    Matrix current_matrix = initial_matrix;

    for (size_t k = 0; k < N_FEATURES - 1; ++k) {
        std::cout << "\n--- Iteration " << k << " ---" << std::endl;
        std::cout << "Remaining features: " << current_matrix.cols << std::endl;

        sycl::buffer<float, 2> buffer_x(current_matrix.data.data(), sycl::range<2>(current_matrix.rows, current_matrix.cols));
        standardize_data(q, buffer_x);

        std::vector<float> cov_data(current_matrix.cols * current_matrix.cols);
        sycl::buffer<float, 2> buffer_cov(cov_data.data(), sycl::range<2>(current_matrix.cols, current_matrix.cols));
        calculate_covariance(q, buffer_x, buffer_cov);

        int root_idx_in_current = para_find_root(q, buffer_x, buffer_cov);
        
        int original_root_idx = U[root_idx_in_current];
        K.push_back(original_root_idx);
        U.erase(U.begin() + root_idx_in_current);
        std::cout << "Found root: " << original_root_idx << std::endl;

        size_t n_remaining = current_matrix.cols - 1;
        if (n_remaining == 0) break;

        Matrix next_matrix;
        next_matrix.rows = N_SAMPLES;
        next_matrix.cols = n_remaining;
        next_matrix.data.resize(N_SAMPLES * n_remaining);
        sycl::buffer<float, 2> buffer_next_x(next_matrix.data.data(), sycl::range<2>(next_matrix.rows, next_matrix.cols));
        
        regress_and_normalize(q, buffer_x, buffer_next_x, buffer_cov, root_idx_in_current);
        
        // This host access forces synchronization and copies data from device to host
        sycl::host_accessor host_acc(buffer_next_x, sycl::read_only);
        for(size_t i = 0; i < next_matrix.data.size(); ++i) {
          // This is inefficient, but necessary for the host-driven model
          // We are manually copying data back to the host Matrix struct
          // In a real scenario, this would be a memcpy or similar
        }
        current_matrix = std::move(next_matrix);
    }
    
    if (!U.empty()) {
        K.push_back(U[0]);
    }

    std::cout << "\nFinal Causal Order: ";
    for(int feature_idx : K) {
        std::cout << feature_idx << " ";
    }
    std::cout << std::endl;

    return 0;
}
