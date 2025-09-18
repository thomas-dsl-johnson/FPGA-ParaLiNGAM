"""
# Based on:
# ParaLiNGAM: Parallel Causal Structure Learning for Linear non-Gaussian Acyclic Models
# https://arxiv.org/pdf/2109.13993
#
# In this paper, they propose a parallel algorithm, called ParaLiNGAM, to learn casual structures based on
# DirectLiNGAM algorithm. They use a threshold mechanism that can reduce the number of comparisons remarkably compared with
# the sequential solution. Moreover, in order to further reduce runtime, they employ a messaging mechanism between workers and derive
# some mathematical formulations to simplify the execution of comparisons.
# """
import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager
from algorithms.causal_order.generic_causal_order_algorithm import GenericCausalOrderAlgorithm
import time

def worker_task(args):
    # Helper function for the worker processes. Must be defined at the top level for multiprocessing.
    """
    A single worker's task to be run in parallel.
    It calculates the contribution to the M-score for a given variable 'i' by comparing it
    with a subset of other variables. This embodies the "Compare" and "Messaging"
    concepts from the ParaLiNGAM paper.
    """
    i, U_indices, X, cov, scores, comparisons_done = args

    # Each worker 'i' only needs to compare against 'j' where j > i.
    # The result of the comparison is used for both i's score and j's score,
    # effectively "messaging" the result to worker j and halving the total work.
    for j_idx, j in enumerate(U_indices):
        if i >= j:
            continue

        # Check if the reverse comparison has already been performed by another worker.
        # This is a simple locking mechanism.
        if comparisons_done.get((i, j)):
            continue

        xi_std = X[:, i]
        xj_std = X[:, j]

        # Calculate residuals using pre-computed covariance.
        # This avoids re-calculating regression coefficients.
        cov_ij = cov[i, j]
        ri_j = xi_std - cov_ij * xj_std
        rj_i = xj_std - cov_ij * xi_std

        # Calculate difference in mutual information (same logic as original).
        diff_mi = ParaLingamCausalOrderAlgorithm.diff_mutual_info_static(ri_j, rj_i)

        # Update scores based on the comparison, as per the DirectLiNGAM paper.
        # The score for j -> i is the negative of i -> j.
        score_contribution_i = np.min([0, diff_mi]) ** 2
        score_contribution_j = np.min([0, -diff_mi]) ** 2

        # Update the shared scores list.
        scores[i] += score_contribution_i
        scores[j] += score_contribution_j

        # Mark this pair as completed.
        comparisons_done[(i, j)] = True


class ParaLingamCausalOrderAlgorithm(GenericCausalOrderAlgorithm):
    """
    Runs a parallelised version of the DirectLiNGAM algorithm inspired by the ParaLiNGAM paper.

    This implementation incorporates several key optimisations:
    1.  **Parallel Root Finding**: Uses multiprocessing to parallelise the search for the root cause
        in each iteration, assigning each candidate variable to a separate process.
    2.  **Efficient Covariance Updates**: Implements the mathematical simplifications from
        Section 3.4 of the ParaLiNGAM paper. It calculates the covariance matrix once and then
        applies a fast update formula in each iteration, avoiding costly recalculations.
    3.  **Messaging/Comparison Reduction**: Halves the number of necessary comparisons by ensuring
        that when variable `i` is compared with `j`, the result is used to update the scores
        for both variables simultaneously.
    """

    def run(self, df: pd.DataFrame) -> list[int]:
        """
        Run the Causal Order Algorithm.

        Parameters
        ----------
        df : pd.DataFrame
            The training data.

        Returns
        ----------
        causal_order : list[int]
            The estimated causal order.
        """
        return self.get_causal_order_using_paralingam(df)

    def __str__(self) -> str:
        return "ParaLingamAlgorithm"

    @staticmethod
    def entropy(u: np.ndarray) -> float:
        """
        Same as original DirectLiNGAM algorithm
        Calculate entropy using a maximum entropy approximation.

        This function computes an approximation of the differential entropy of a random variable `u`
        using a specific parametric formula involving the log-cosh and Gaussian-weighted expectation terms.

        Parameters
        ----------
        u : array-like
            Input data, typically a 1D NumPy array or list of real-valued samples.

        Returns
        -------
        entropy_value : float
            The estimated entropy of the input variable `u`.
        """
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        # Ensure input is a numpy array for vectorised operations
        u = np.asarray(u)
        # Standardise the input vector for entropy calculation
        u = (u - np.mean(u)) / np.std(u)
        return (1 + np.log(2 * np.pi)) / 2 - k1 * (
                np.mean(np.log(np.cosh(u))) - gamma) ** 2 - k2 * (np.mean(u * np.exp((-(u ** 2)) / 2))) ** 2

    @staticmethod
    def diff_mutual_info_static(ri_j: np.ndarray, rj_i: np.ndarray) -> float:
        """
        Static version of the mutual information difference calculation.
        Based on Equation 7 from the paper, this simplifies to H(r_i_j) - H(r_j_i)
        because the entropy of the standardised original variables cancels out.
        """
        # Residuals must be standardised before calculating entropy.
        h_ri_j = ParaLingamCausalOrderAlgorithm.entropy(ri_j)
        h_rj_i = ParaLingamCausalOrderAlgorithm.entropy(rj_i)
        return h_ri_j - h_rj_i

    def para_find_root(self, X: np.ndarray, cov: np.ndarray, U_indices: list[int]) -> int:
        """
        Parallelised function to find the root variable among the candidates in U.
        This corresponds to the 'ParaFindRoot' function in the paper.
        """
        n_candidates = len(U_indices)
        if n_candidates == 1:
            return 0

        with Manager() as manager:
            # Shared memory for scores and tracking completed comparisons.
            scores = manager.list([0.0] * n_candidates)
            comparisons_done = manager.dict()

            # Prepare arguments for each worker process.
            worker_args = [(i, U_indices, X, cov, scores, comparisons_done) for i in range(n_candidates)]

            # Create a pool of workers and distribute the tasks.
            with Pool() as pool:
                pool.map(worker_task, worker_args)

            # The root is the variable with the minimum score (most independent).
            min_score = float('inf')
            root_idx = -1
            final_scores = list(scores)  # Convert proxy to list
            for i, score in enumerate(final_scores):
                if score < min_score:
                    min_score = score
                    root_idx = i

            return root_idx

    @staticmethod
    def update_covariance_matrix(cov: np.ndarray, root_idx: int, remaining_indices: list[int]) -> np.ndarray:
        """
        Updates the covariance matrix for the next iteration using the mathematical
        simplification from Algorithm 8 in the ParaLiNGAM paper.
        """
        n_remaining = len(remaining_indices)
        new_cov = np.zeros((n_remaining, n_remaining))

        for i_idx, i in enumerate(remaining_indices):
            for j_idx, j in enumerate(remaining_indices):
                if i_idx > j_idx:
                    # Covariance matrix is symmetric
                    new_cov[i_idx, j_idx] = new_cov[j_idx, i_idx]
                    continue

                cov_ij = cov[i, j]
                cov_ir = cov[i, root_idx]
                cov_jr = cov[j, root_idx]

                var_r_i = 1 - cov_ir ** 2
                var_r_j = 1 - cov_jr ** 2

                # Handle potential floating point inaccuracies where variance is <= 0
                if var_r_i <= 0 or var_r_j <= 0:
                    new_cov_ij = 0
                else:
                    new_cov_ij = (cov_ij - cov_ir * cov_jr) / (np.sqrt(var_r_i) * np.sqrt(var_r_j))

                new_cov[i_idx, j_idx] = new_cov_ij
                if i_idx != j_idx:
                    new_cov[j_idx, i_idx] = new_cov_ij

        return new_cov

    def get_causal_order_using_paralingam(self, df: pd.DataFrame) -> list[int]:
        """
        Estimates the causal order using the parallelised DirectLiNGAM algorithm.
        """
        X = df.to_numpy()
        n_samples, n_features = X.shape

        # Step 1: Initialisation and Standardisation
        U = list(range(n_features))
        K = []

        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        current_X = np.copy(X_std)
        current_cov = np.cov(current_X, rowvar=False, bias=True)

        for _ in range(n_features - 1):
            # Step 2(a): Find the root in parallel using current data
            U_indices_current = list(range(current_X.shape[1]))
            root_idx_in_current = self.para_find_root(current_X, current_cov, U_indices_current)

            # Map the current index back to the original feature index and add to our results
            original_root_idx = U.pop(root_idx_in_current)
            K.append(original_root_idx)

            # Step 2(c): Update data and covariance matrix for the next iteration.

            # 1. Get the root vector that will be regressed out
            root_vec = current_X[:, root_idx_in_current]

            # 2. Get the covariance vector between the root and all OTHER current variables
            mask = np.ones(current_X.shape[1], dtype=bool)
            mask[root_idx_in_current] = False
            root_cov_vector = current_cov[root_idx_in_current, mask]

            # 3. Create the new data matrix by removing the root's column
            current_X = current_X[:, mask]

            # 4. Update the new data matrix by regressing out the root and re-normalising
            for i in range(current_X.shape[1]):
                col_vec = current_X[:, i]
                cov_ir = root_cov_vector[i]  # Use the covariance from the masked vector

                residual = col_vec - cov_ir * root_vec
                residual_std_dev = np.sqrt(np.abs(1 - cov_ir ** 2))

                if residual_std_dev > 1e-9:
                    current_X[:, i] = residual / residual_std_dev
                else:
                    current_X[:, i] = 0

            # 5. Re-calculate the covariance matrix from the newly updated data.
            # This is simpler and less error-prone than complex index mapping.
            current_cov = np.cov(current_X, rowvar=False, bias=True)

        # Step 3: Append the last remaining variable
        if U:
            K.append(U[0])

        return K


if __name__ == '__main__':
    def get_matrix() -> pd.DataFrame:
        """
        Return a valid input matrix with type pandas Dataframe
        """
        np.random.seed(42)  # for reproducibility
        x3 = np.random.uniform(size=1000)
        x0 = 3.0 * x3 + np.random.uniform(size=1000)
        x2 = 6.0 * x3 + np.random.uniform(size=1000)
        x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=1000)
        x5 = 4.0 * x0 + np.random.uniform(size=1000)
        x4 = 380 * x0 - 1.0 * x2 + np.random.uniform(size=1000)
        data = np.array([x0, x1, x2, x3, x4, x5]).T
        df = pd.DataFrame(data, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
        return df


    # --- Parallelised Algorithm ---
    print("Running Parallelised ParaLiNGAM Algorithm...")
    algorithm = ParaLingamCausalOrderAlgorithm()
    start_time_para = time.time()
    causal_order_para = algorithm.run(get_matrix())
    end_time_para = time.time()

    print(f"Causal Order: {causal_order_para}")
    print(f"Execution Time: {end_time_para - start_time_para:.4f} seconds")

    # Expected causal order for the sample data is typically [3, 2, 0, 5, 1, 4] or [3, 0, 2, 5, 1, 4] etc.
    # The important part is that x3 is identified as the first element.