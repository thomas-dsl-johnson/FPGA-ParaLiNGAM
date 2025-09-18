import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager
import time
import sys
import os


# This worker task remains the same as the faithful implementation.
def worker_task(args):
    """
    A single worker's task to be run in parallel. It calculates the scores for a
    given variable by comparing it with all other variables.
    """
    i, U_indices, X, cov, scores, comparisons_done = args
    n_samples = X.shape[0]

    for j_idx, j in enumerate(U_indices):
        # Use a simple locking mechanism to ensure a pair is only compared once.
        pair = tuple(sorted((i, j)))
        if i == j or comparisons_done.get(pair):
            continue

        xi_std = X[:, i]
        xj_std = X[:, j]

        cov_ij = cov[i, j]
        ri_j = xi_std - cov_ij * xj_std
        rj_i = xj_std - cov_ij * xi_std

        # Entropy calculation is nested here for clarity
        def entropy(u: np.ndarray) -> float:
            k1 = 79.047
            k2 = 7.4129
            gamma = 0.37457
            u = (u - np.mean(u)) / np.std(u)
            term1 = np.mean(np.log(np.cosh(u))) - gamma
            term2 = np.mean(u * np.exp(-0.5 * u ** 2))
            return (1 + np.log(2 * np.pi)) / 2 - k1 * (term1 ** 2) - k2 * (term2 ** 2)

        diff_mi = entropy(ri_j) - entropy(rj_i)

        score_contribution_i = min(0, diff_mi) ** 2
        score_contribution_j = min(0, -diff_mi) ** 2

        # Atomically update scores and mark comparison as done
        scores[i] += score_contribution_i
        scores[j] += score_contribution_j
        comparisons_done[pair] = True


class ParaLingamCausalOrderAlgorithm:
    """
    A faithful Python implementation mirroring the logic of the final DPC++ code.
    It uses the efficient covariance update but omits thresholding.
    """

    def _update_covariance_matrix(self, cov: np.ndarray, root_idx: int) -> np.ndarray:
        """
        Updates the covariance matrix using the efficient formula from the paper.
        """
        remaining_indices = list(range(cov.shape[0]))
        remaining_indices.pop(root_idx)

        if not remaining_indices:
            return np.array([])

        # Select the submatrix of remaining variables
        sub_cov = cov[np.ix_(remaining_indices, remaining_indices)]

        cov_ir = cov[remaining_indices, root_idx]

        # Numerically robust variance calculation for residuals
        var_r_i = np.abs(1.0 - cov_ir ** 2)

        # Create a normalization matrix from the outer product of std_devs
        std_devs = np.sqrt(var_r_i)
        norm_matrix = np.outer(std_devs, std_devs)

        # Avoid division by zero
        norm_matrix[norm_matrix < 1e-9] = 1.0

        # Calculate the new covariance matrix with vectorized operations
        new_cov = (sub_cov - np.outer(cov_ir, cov_ir)) / norm_matrix
        np.fill_diagonal(new_cov, 1.0)

        return new_cov

    def _para_find_root(self, X: np.ndarray, cov: np.ndarray) -> int:
        """
        Parallelised function to find the root variable. Mirrors the "Scatter-Reduce"
        logic of the DPC++ implementation.
        """
        n_candidates = X.shape[1]
        if n_candidates <= 1:
            return 0

        with Manager() as manager:
            scores = manager.list([0.0] * n_candidates)
            comparisons_done = manager.dict()

            worker_args = [
                (i, list(range(n_candidates)), X, cov, scores, comparisons_done)
                for i in range(n_candidates)
            ]

            with Pool() as pool:
                pool.map(worker_task, worker_args)

            # Find the root with the minimum score on the host
            return np.argmin(list(scores))

    def run(self, df: pd.DataFrame) -> list[int]:
        """
        Estimates the causal order using the parallelised algorithm.
        """
        X = df.to_numpy()
        n_features = X.shape[1]

        U = list(range(n_features))
        K = []

        # Initial standardization and covariance calculation
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)
        current_X = np.copy(X_std)
        current_cov = np.cov(current_X, rowvar=False, bias=True)

        for _ in range(n_features - 1):
            root_idx_in_current = self._para_find_root(current_X, current_cov)

            original_root_idx = U.pop(root_idx_in_current)
            K.append(original_root_idx)

            # Update data by regressing out the root's effect
            root_vec = current_X[:, root_idx_in_current].copy()
            mask = np.ones(current_X.shape[1], dtype=bool)
            mask[root_idx_in_current] = False

            remaining_X = current_X[:, mask]
            cov_vector = current_cov[root_idx_in_current, mask]

            residual_std_devs = np.sqrt(np.abs(1 - cov_vector ** 2))
            residual_std_devs[residual_std_devs < 1e-9] = 1.0

            remaining_X -= root_vec[:, np.newaxis] * cov_vector
            remaining_X /= residual_std_devs

            current_X = remaining_X

            # EFFICIENT COVARIANCE UPDATE
            current_cov = self._update_covariance_matrix(current_cov, root_idx_in_current)

        if U:
            K.append(U[0])

        return K


# Generates the same sample data as the C++ example.
def get_matrix() -> pd.DataFrame:
    """
    Return a valid input matrix with type pandas Dataframe
    """
    print("Generating sample matrix...")
    np.random.seed(42)
    x3 = np.random.uniform(size=1000)
    x0 = 3.0 * x3 + np.random.uniform(size=1000)
    x2 = 6.0 * x3 + np.random.uniform(size=1000)
    x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=1000)
    x5 = 4.0 * x0 + np.random.uniform(size=1000)
    x4 = 8.0 * x0 - 1.0 * x2 + np.random.uniform(size=1000)
    data = np.array([x0, x1, x2, x3, x4, x5]).T
    df = pd.DataFrame(data, columns=[f'x{i}' for i in range(6)])
    return df


# Reads a CSV file into a pandas DataFrame.
def read_csv(filepath: str) -> pd.DataFrame:
    """
    Reads a CSV file, skipping any header row.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Reading data from {filepath}...")
    # Assume the CSV may or may not have a header
    df = pd.read_csv(filepath, header=None)
    # Drop the first column (e.g., a timestamp or index column)
    df = df.iloc[:, 1:]
    # **FIX APPLIED HERE**: Convert all columns to numeric types.
    # Any non-numeric values (like headers) will become NaN.
    df = df.apply(pd.to_numeric, errors='coerce')
    # Replace any NaN values with 0 to prevent errors in calculations.
    df = df.fillna(0)
    print(f"Successfully read and cleaned {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


if __name__ == '__main__':
    print("Running FAITHFUL Python ParaLiNGAM Algorithm...")

    algorithm = ParaLingamCausalOrderAlgorithm()
    df = None

    try:
        # Check for a command-line argument (the CSV file path)
        if len(sys.argv) > 1:
            filepath = sys.argv[1]
            df = read_csv(filepath)
        else:
            # If no CSV is provided, fall back to the generated matrix
            print("No CSV file provided. Falling back to sample data generator.")
            df = get_matrix()

        start_time = time.time()
        causal_order = algorithm.run(df)
        end_time = time.time()

        print(f"\nCausal Order: {causal_order}")
        print(f"Execution Time: {end_time - start_time:.4f} seconds")

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)

