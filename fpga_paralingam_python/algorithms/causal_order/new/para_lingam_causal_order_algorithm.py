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
    """
    A single worker's task to be run in parallel.
    It calculates the contribution to the M-score for a given variable 'i' by comparing it
    with a subset of other variables. This embodies the "Compare", "Messaging",
    and "Threshold" concepts from the ParaLiNGAM paper[cite: 217, 218, 227].

    A worker will stop its comparisons if its score exceeds the shared threshold[cite: 315].
    """
    i, U_indices, X, cov, scores, comparisons_done, threshold, worker_status = args

    # Each worker 'i' only needs to compare against 'j' where j > i.
    # The result of the comparison is used for both i's score and j's score,
    # effectively "messaging" the result and halving the total work[cite: 297].
    for j_idx, j in enumerate(U_indices):
        if i >= j:
            continue

        # If score exceeds threshold, stop performing comparisons for this worker[cite: 310, 443].
        # This is the core of the threshold mechanism.
        if scores[i] > threshold.value:
            worker_status[i] = 'threshold_exceeded'
            return

        # Use a simple locking mechanism to ensure a pair is only compared once.
        pair = tuple(sorted((i, j)))
        if comparisons_done.get(pair):
            continue

        xi_std = X[:, i]
        xj_std = X[:, j]

        # Calculate residuals using pre-computed covariance.
        # This aligns with the paper's note that residuals only need the covariance matrix[cite: 361].
        cov_ij = cov[i, j]
        ri_j = xi_std - cov_ij * xj_std
        rj_i = xj_std - cov_ij * xi_std

        # Calculate difference in mutual information (same logic as original).
        diff_mi = ParaLingamCausalOrderAlgorithmNew.diff_mutual_info_static(ri_j, rj_i)

        # Update scores based on the comparison, as per the DirectLiNGAM paper.
        # The score for j -> i is the negative of i -> j.
        score_contribution_i = np.min([0, diff_mi]) ** 2
        score_contribution_j = np.min([0, -diff_mi]) ** 2

        # Update the shared scores list.
        scores[i] += score_contribution_i
        scores[j] += score_contribution_j

        # Mark this pair as completed.
        comparisons_done[pair] = True

    # If the loop completes, the worker has finished all its comparisons.
    worker_status[i] = 'completed'


class ParaLingamCausalOrderAlgorithmNew(GenericCausalOrderAlgorithm):
    """
    Runs a parallelised version of the DirectLiNGAM algorithm inspired by the ParaLiNGAM paper.

    This implementation incorporates several key optimisations:
    1.  **Parallel Root Finding**: Uses multiprocessing to parallelise the search for the root cause
        in each iteration, assigning each candidate variable to a separate process[cite: 72].
    2.  **Threshold Mechanism**: Implements the thresholding and scheduling logic from the paper
        to dramatically reduce the number of pairwise comparisons by terminating workers whose
        scores exceed a dynamic threshold[cite: 10, 311].
    3.  **Efficient Covariance Updates**: Implements the mathematical simplifications from
        Section 3.4 of the ParaLiNGAM paper[cite: 290]. It calculates the covariance matrix once and then
        applies a fast update formula in each iteration, avoiding costly recalculations[cite: 248, 387].
    4.  **Messaging/Comparison Reduction**: Halves the number of necessary comparisons by ensuring
        that when variable `i` is compared with `j`, the result is used to update the scores
        for both variables simultaneously[cite: 11, 296, 297].
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
        Calculate entropy using a maximum entropy approximation[cite: 200, 201].
        """
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        u = np.asarray(u)
        u = (u - np.mean(u)) / np.std(u)
        return (1 + np.log(2 * np.pi)) / 2 - k1 * (
                np.mean(np.log(np.cosh(u))) - gamma) ** 2 - k2 * (np.mean(u * np.exp((-(u ** 2)) / 2))) ** 2

    @staticmethod
    def diff_mutual_info_static(ri_j: np.ndarray, rj_i: np.ndarray) -> float:
        """
        Static version of the mutual information difference calculation.
        Based on Equation 7 from the paper, this simplifies to H(r_i_j) - H(r_j_i)
        because the entropy of the standardised original variables cancels out[cite: 199].
        """
        h_ri_j = ParaLingamCausalOrderAlgorithmNew.entropy(ri_j)
        h_rj_i = ParaLingamCausalOrderAlgorithmNew.entropy(rj_i)
        return h_ri_j - h_rj_i

    def para_find_root(self, X: np.ndarray, cov: np.ndarray, U_indices: list[int]) -> int:
        """
        Parallelised function to find the root variable among the candidates in U.
        This corresponds to the 'ParaFindRoot' function and incorporates the iterative
        scheduling and thresholding logic from the paper[cite: 220, 331].
        """
        n_candidates = len(U_indices)
        if n_candidates == 1:
            return 0

        with Manager() as manager:
            # Shared memory for scores and tracking.
            scores = manager.list([0.0] * n_candidates)
            comparisons_done = manager.dict()
            # The threshold 'gamma' starts small and is increased if needed[cite: 314].
            threshold = manager.Value('f', 0.01)
            # Tracks if a worker finished all comparisons or was stopped by the threshold.
            worker_status = manager.list(['pending'] * n_candidates)

            finished = False
            while not finished:
                worker_args = [
                    (i, U_indices, X, cov, scores, comparisons_done, threshold, worker_status)
                    for i in range(n_candidates)
                ]

                with Pool() as pool:
                    pool.map(worker_task, worker_args)

                # --- Scheduler Logic (based on Algorithm 6) ---
                # Check if termination condition is met: at least one worker completed
                # its comparisons with a score below the threshold[cite: 332].
                can_terminate = any(
                    status == 'completed' and scores[i] <= threshold.value
                    for i, status in enumerate(worker_status)
                )

                if can_terminate:
                    finished = True
                else:
                    # If all active workers have scores above the threshold, increase it[cite: 316, 338].
                    all_workers_stuck = all(score > threshold.value for score in scores)
                    if all_workers_stuck:
                        threshold.value *= 2  # Simple update rule [cite: 340]

                    # Reset status for workers that can continue with the new threshold
                    for i in range(n_candidates):
                        if worker_status[i] == 'threshold_exceeded':
                            worker_status[i] = 'pending'

            # Find the root among the workers that completed below the threshold.
            min_score = float('inf')
            root_idx = -1
            final_scores = list(scores)
            final_status = list(worker_status)

            for i, score in enumerate(final_scores):
                # The root is the worker with the minimum score from the group that
                # finished its comparisons below the final threshold[cite: 323, 324].
                if final_status[i] == 'completed' and score <= threshold.value:
                    if score < min_score:
                        min_score = score
                        root_idx = i

            # This case should ideally not be hit if the threshold logic works correctly,
            # but as a fallback, we pick the absolute minimum score.
            if root_idx == -1:
                root_idx = np.argmin(final_scores)

            return root_idx

    @staticmethod
    def update_covariance_matrix(cov: np.ndarray, root_idx: int) -> np.ndarray:
        """
        Updates the covariance matrix for the next iteration using the mathematical
        simplification from Algorithm 8 in the ParaLiNGAM paper[cite: 389].
        This is much faster than re-computing from the data.
        """
        # Get indices for remaining variables
        remaining_indices = list(range(cov.shape[0]))
        remaining_indices.pop(root_idx)
        n_remaining = len(remaining_indices)

        if n_remaining == 0:
            return np.array([])

        new_cov = np.zeros((n_remaining, n_remaining))

        for i_idx_new, i_idx_old in enumerate(remaining_indices):
            for j_idx_new, j_idx_old in enumerate(remaining_indices):
                if i_idx_new > j_idx_new:
                    continue  # Symmetric matrix

                cov_ij = cov[i_idx_old, j_idx_old]
                cov_ir = cov[i_idx_old, root_idx]
                cov_jr = cov[j_idx_old, root_idx]

                # Variance of residuals: var(r_i) = 1 - cov(x_i, x_root)^2 [cite: 365]
                var_r_i = 1 - cov_ir ** 2
                var_r_j = 1 - cov_jr ** 2

                if var_r_i <= 1e-9 or var_r_j <= 1e-9:
                    new_cov_ij = 0
                else:
                    # Updated covariance formula from paper[cite: 386, 387].
                    new_cov_ij = (cov_ij - cov_ir * cov_jr) / (np.sqrt(var_r_i) * np.sqrt(var_r_j))

                new_cov[i_idx_new, j_idx_new] = new_cov_ij
                if i_idx_new != j_idx_new:
                    new_cov[j_idx_new, i_idx_new] = new_cov_ij

        # Ensure diagonal is 1.0 for standardized data
        np.fill_diagonal(new_cov, 1.0)
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
            # Step 2(a): Find the root in parallel using current data and covariance
            U_indices_current = list(range(current_X.shape[1]))
            root_idx_in_current = self.para_find_root(current_X, current_cov, U_indices_current)

            original_root_idx = U.pop(root_idx_in_current)
            K.append(original_root_idx)

            # Step 2(c): Update data and covariance matrix for the next iteration.

            # 1. Update the data matrix by regressing out the root's effect (Algorithm 7) [cite: 370]
            root_vec = current_X[:, root_idx_in_current].copy()  # Make a copy
            mask = np.ones(current_X.shape[1], dtype=bool)
            mask[root_idx_in_current] = False

            remaining_X = current_X[:, mask]
            cov_vector = current_cov[root_idx_in_current, mask]

            for i in range(remaining_X.shape[1]):
                col_vec = remaining_X[:, i]
                cov_ir = cov_vector[i]

                residual = col_vec - cov_ir * root_vec
                # Denominator is sqrt of residual variance [cite: 367]
                residual_std_dev = np.sqrt(np.abs(1 - cov_ir ** 2))

                if residual_std_dev > 1e-9:
                    remaining_X[:, i] = residual / residual_std_dev
                else:
                    remaining_X[:, i] = 0

            current_X = remaining_X

            # 2. Update the covariance matrix using the efficient formula (Algorithm 8) [cite: 248]
            # This is much faster than recalculating with np.cov().
            current_cov = self.update_covariance_matrix(current_cov, root_idx_in_current)

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
    print("Running Revised and Corrected ParaLiNGAM Algorithm...")
    algorithm = ParaLingamCausalOrderAlgorithmNew()
    start_time_para = time.time()
    causal_order_para = algorithm.run(get_matrix())
    end_time_para = time.time()

    print(f"Causal Order: {causal_order_para}")
    print(f"Execution Time: {end_time_para - start_time_para:.4f} seconds")

    # Expected causal order for the sample data is typically [3, 2, 0, 5, 1, 4] or [3, 0, 2, 5, 1, 4] etc.
    # The important part is that x3 is identified as the first element.