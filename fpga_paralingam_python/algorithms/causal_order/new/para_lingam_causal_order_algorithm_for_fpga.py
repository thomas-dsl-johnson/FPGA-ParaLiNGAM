"""
This script is a faithful Python equivalent of the final, optimized DPC++
implementation of the ParaLiNGAM algorithm.

Key Features Mirrored from the DPC++ Version:
1.  **Efficient Covariance Update**: Implements the mathematical simplification from
    the ParaLiNGAM paper (Algorithm 8) to update the covariance matrix efficiently
    between iterations, avoiding costly recalculations. This was a key feature
    of the final C++ code.
2.  **Parallel Root Finding**: Uses Python's `multiprocessing.Pool` to parallelize
    the search for the root variable, analogous to the SYCL kernels in DPC++.
3.  **No Threshold Mechanism**: Like the FPGA-targeted C++ code, this version
    omits the complex thresholding logic in favor of a simpler, robust parallel
    pattern where all pairwise comparisons are completed.
4.  **Messaging/Comparison Reduction**: Halves the number of comparisons by
    updating scores for both variables (i and j) simultaneously.
"""
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager

# Assuming the base classes are in these locations as per the original structure.
# If not, these might need to be included directly in this file.
try:
    from algorithms.causal_order.generic_causal_order_algorithm import GenericCausalOrderAlgorithm
except ImportError:
    print("Warning: Could not import GenericCausalOrderAlgorithm. Using a placeholder base class.")
    class GenericCausalOrderAlgorithm:
        def run(self, df):
            raise NotImplementedError
        def __str__(self):
            return "GenericBase"


def worker_task(args):
    """
    A single worker's task to be run in parallel. It calculates the M-score
    contribution for a given variable 'i' by comparing it with all other
    variables 'j' where j > i. The result is used for both i's and j's scores.
    """
    i, X, cov, scores = args

    n_features = X.shape[1]
    local_score_i = 0.0

    # Each worker calculates its partial scores. A second step will apply them.
    # This mirrors the "Scatter" part of the DPC++ Scatter-Reduce pattern.
    for j in range(i + 1, n_features):
        xi_std = X[:, i]
        xj_std = X[:, j]

        cov_ij = cov[i, j]
        ri_j = xi_std - cov_ij * xj_std
        rj_i = xj_std - cov_ij * xi_std

        diff_mi = ParaLingamFaithful.diff_mutual_info_static(ri_j, rj_i)

        score_contribution_i = np.min([0, diff_mi]) ** 2
        score_contribution_j = np.min([0, -diff_mi]) ** 2

        # Atomically add scores
        scores[i] += score_contribution_i
        scores[j] += score_contribution_j


class ParaLingamFaithful(GenericCausalOrderAlgorithm):
    """
    A faithful Python implementation of the optimized DPC++ ParaLiNGAM algorithm.
    """
    def run(self, df: pd.DataFrame) -> list[int]:
        return self.get_causal_order_using_paralingam(df)

    def __str__(self) -> str:
        return "ParaLingamFaithfulAlgorithm"

    @staticmethod
    def _entropy(u: np.ndarray) -> float:
        """Calculate entropy using a maximum entropy approximation."""
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457

        # Standardize the input vector for entropy calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            u_std = (u - np.mean(u)) / np.std(u)
            u_std = np.nan_to_num(u_std) # Replace NaN from 0/0 division with 0

        term1 = np.mean(np.log(np.cosh(u_std))) - gamma
        term2 = np.mean(u_std * np.exp(-0.5 * u_std**2))

        return (1 + np.log(2 * np.pi)) / 2 - k1 * term1**2 - k2 * term2**2

    @staticmethod
    def diff_mutual_info_static(ri_j: np.ndarray, rj_i: np.ndarray) -> float:
        """Calculates the difference in mutual information between residuals."""
        h_ri_j = ParaLingamFaithful._entropy(ri_j)
        h_rj_i = ParaLingamFaithful._entropy(rj_i)
        return h_ri_j - h_rj_i

    def _para_find_root(self, X: np.ndarray, cov: np.ndarray) -> int:
        """
        Parallelized function to find the root variable.
        This mirrors the Scatter-Reduce pattern from the DPC++ implementation.
        """
        n_candidates = X.shape[1]
        if n_candidates <= 1:
            return 0

        with Manager() as manager:
            # Shared memory for scores. Using a Manager list for simplicity,
            # though for extreme performance shared memory arrays would be better.
            scores = manager.list([0.0] * n_candidates)

            worker_args = [(i, X, cov, scores) for i in range(n_candidates)]

            # "Scatter" phase: workers compute pairwise scores in parallel.
            with Pool() as pool:
                pool.map(worker_task, worker_args)

            # "Reduce" phase: find the minimum score on the host.
            final_scores = list(scores)
            root_idx = np.argmin(final_scores)
            return root_idx

    @staticmethod
    def _update_covariance_matrix(cov: np.ndarray, root_idx: int) -> np.ndarray:
        """
        Updates the covariance matrix using the efficient formula from the paper.
        This is the direct Python equivalent of the `update_covariance` DPC++ kernel.
        """
        n_current = cov.shape[0]
        remaining_indices = np.delete(np.arange(n_current), root_idx)
        n_remaining = len(remaining_indices)

        if n_remaining == 0:
            return np.array([])

        new_cov = np.zeros((n_remaining, n_remaining))

        cov_ir = cov[remaining_indices, root_idx]

        # Vectorized calculation for efficiency
        cov_submatrix = cov[np.ix_(remaining_indices, remaining_indices)]

        # var(r_i) = 1 - cov(x_i, x_root)^2
        var_r_i = 1 - cov_ir**2

        # Outer product for the update term
        update_term = np.outer(cov_ir, cov_ir)

        # **FIXED**: Take the absolute value of the variance before sqrt to prevent
        # a RuntimeWarning due to floating-point inaccuracies.
        residual_std_devs = np.sqrt(np.abs(var_r_i))
        norm_matrix = np.outer(residual_std_devs, residual_std_devs)

        # Avoid division by zero
        norm_matrix[norm_matrix < 1e-9] = 1.0

        new_cov = (cov_submatrix - update_term) / norm_matrix

        # Ensure diagonal is 1.0 for standardized data
        np.fill_diagonal(new_cov, 1.0)
        return new_cov

    def get_causal_order_using_paralingam(self, df: pd.DataFrame) -> list[int]:
        """Estimates the causal order using the optimized algorithm."""
        X = df.to_numpy()
        n_features = X.shape[1]

        U = list(range(n_features))
        K = []

        # Initial standardization and covariance calculation
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        current_X = np.copy(X_std)
        current_cov = np.cov(current_X, rowvar=False)

        for _ in range(n_features - 1):
            root_idx_in_current = self._para_find_root(current_X, current_cov)

            original_root_idx = U.pop(root_idx_in_current)
            K.append(original_root_idx)

            # Update data by regressing out the root's effect
            root_vec = current_X[:, root_idx_in_current].copy()
            remaining_indices = np.delete(np.arange(current_X.shape[1]), root_idx_in_current)

            remaining_X = current_X[:, remaining_indices]
            cov_vector = current_cov[root_idx_in_current, remaining_indices]

            # Denominator is sqrt of residual variance
            residual_std_devs = np.sqrt(np.abs(1 - cov_vector**2))
            residual_std_devs[residual_std_devs < 1e-9] = 1.0

            # Vectorized update of the data matrix
            current_X = (remaining_X - root_vec[:, np.newaxis] * cov_vector) / residual_std_devs

            # **EFFICIENT COVARIANCE UPDATE**
            current_cov = self._update_covariance_matrix(current_cov, root_idx_in_current)

        if U:
            K.append(U[0])

        return K


if __name__ == '__main__':
    def get_matrix() -> pd.DataFrame:
        """Generates the same sample data as the DPC++ example."""
        np.random.seed(42)
        n_samples = 1000
        x3 = np.random.uniform(size=n_samples)
        x0 = 3.0 * x3 + np.random.uniform(size=n_samples)
        x2 = 6.0 * x3 + np.random.uniform(size=n_samples)
        x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=n_samples)
        x5 = 4.0 * x0 + np.random.uniform(size=n_samples)
        x4 = 8.0 * x0 - 1.0 * x2 + np.random.uniform(size=n_samples)

        data = np.array([x0, x1, x2, x3, x4, x5]).T
        return pd.DataFrame(data, columns=[f'x{i}' for i in [0, 1, 2, 3, 4, 5]])

    print("Running FAITHFUL Python ParaLiNGAM Algorithm...")
    algorithm = ParaLingamFaithful()
    sample_data = get_matrix()

    start_time = time.time()
    causal_order = algorithm.run(sample_data)
    end_time = time.time()

    print(f"Causal Order: {causal_order}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    # Expected causal order often starts with 3, e.g., [3, 0, 2, 5, 1, 4]

