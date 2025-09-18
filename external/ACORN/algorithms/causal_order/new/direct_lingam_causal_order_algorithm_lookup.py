import pandas as pd
import numpy as np
from algorithms.causal_order.generic_causal_order_algorithm import GenericCausalOrderAlgorithm


class DirectLingamCausalOrderAlgorithmLookup(GenericCausalOrderAlgorithm):
    """
    Implements DirectLiNGAM using the 'Mathematical Simplifications' from the
    ParaLiNGAM paper (Section 3.4) for improved performance.

    The key changes are:
    1.  The covariance matrix is computed once per iteration and passed to the
        search function to speed up residual calculations.
    2.  After a root cause is found, the data and covariance matrices for the
        next iteration are calculated using efficient update formulas derived
        in the paper[cite: 365, 386], rather than being re-computed from scratch.
    """

    def run(self, df: pd.DataFrame) -> list[int]:
        """
        Run the Causal Order Algorithm.
        """
        return self.get_causal_order_using_simplifications(df)

    def __str__(self) -> str:
        return "SimplifiedLingamAlgorithm"

    # The static helper methods for entropy and diff_mutual_info are unchanged.
    @staticmethod
    def entropy(u: np.ndarray) -> float:
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        return (1 + np.log(2 * np.pi)) / 2 - k1 * (
                np.mean(np.log(np.cosh(u))) - gamma) ** 2 - k2 * (np.mean(u * np.exp((-(u ** 2)) / 2))) ** 2

    def diff_mutual_info(self, xi_std: np.ndarray, xj_std: np.ndarray, ri_j: np.ndarray, rj_i: np.ndarray) -> float:
        return (self.entropy(xj_std) + self.entropy(ri_j / np.std(ri_j))) - (
                self.entropy(xi_std) + self.entropy(rj_i / np.std(rj_i))
        )

    def simplified_residual(self, xi: np.ndarray, xj: np.ndarray, cov_ij: float) -> np.ndarray:
        """
        Calculates the residual of xi regressed on xj.
        Uses a pre-computed covariance for efficiency, as the variance of a
        standardised variable is 1.
        """
        return xi - cov_ij * xj

    def search_causal_order_simplified(self, X: np.ndarray, cov: np.ndarray, U: list[int]) -> int:
        """
        Search for the next variable in the causal order.
        This version is simplified to accept a pre-computed covariance matrix.
        """
        Uc = U
        if len(Uc) == 1:
            return Uc[0]

        M_list = []
        for i_idx, i in enumerate(Uc):
            M = 0
            for j_idx, j in enumerate(Uc):
                if i == j:
                    continue

                xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])

                # Get the regression coefficient directly from the covariance matrix.
                # This avoids calling np.cov and np.var in this tight inner loop.
                cov_ij = cov[i_idx, j_idx]

                # Calculate residuals using the simplified method
                ri_j = self.simplified_residual(xi_std, xj_std, cov_ij)
                rj_i = self.simplified_residual(xj_std, xi_std, cov_ij)

                M += np.min([0, self.diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
            M_list.append(-1.0 * M)

        xm_idx = np.argmax(M_list)
        return Uc[xm_idx]

    def update_state(self, X: np.ndarray, cov: np.ndarray, root_idx_in_current: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Updates the data and covariance matrices for the next iteration using the
        efficient formulas from the ParaLiNGAM paper.

        Returns:
        A tuple containing the new data matrix and new covariance matrix.
        """
        n_current = X.shape[1]

        # --- Update Data Matrix (implements logic from Algorithm 7) ---
        root_vec = X[:, root_idx_in_current]
        new_X = np.zeros((X.shape[0], n_current - 1))

        # --- Update Covariance Matrix (implements logic from Algorithm 8) ---
        new_cov = np.zeros((n_current - 1, n_current - 1))

        remaining_indices = [i for i in range(n_current) if i != root_idx_in_current]

        # Update Data
        for i, col_idx in enumerate(remaining_indices):
            cov_ir = cov[col_idx, root_idx_in_current]
            variance_of_residual = 1 - cov_ir ** 2
            std_dev_of_residual = np.sqrt(max(variance_of_residual, 1e-12))  # Avoid sqrt of negative

            # Update data column by regressing out the root and re-normalising
            new_X[:, i] = (X[:, col_idx] - cov_ir * root_vec) / std_dev_of_residual

        # Update Covariance
        for i_idx, i in enumerate(remaining_indices):
            for j_idx, j in enumerate(remaining_indices):
                # The new covariance matrix is symmetric
                if i_idx > j_idx:
                    new_cov[i_idx, j_idx] = new_cov[j_idx, i_idx]
                    continue

                cov_ij = cov[i, j]
                cov_ir = cov[i, root_idx_in_current]
                cov_jr = cov[j, root_idx_in_current]

                # Get variance of residuals for normalisation factor
                var_r_i = 1 - cov_ir ** 2
                var_r_j = 1 - cov_jr ** 2

                # Equation 11 divided by standard deviations of residuals
                denominator = np.sqrt(max(var_r_i, 1e-12) * max(var_r_j, 1e-12))
                new_cov[i_idx, j_idx] = (cov_ij - cov_ir * cov_jr) / denominator
                if i_idx != j_idx:
                    new_cov[j_idx, i_idx] = new_cov[i_idx, j_idx]

        return new_X, new_cov

    def get_causal_order_using_simplifications(self, df: pd.DataFrame) -> list[int]:
        """
        Main driver loop that leverages the mathematical simplifications.
        """
        n_features = df.shape[1]

        # Initialise with original data and indices
        X_orig = df.to_numpy()
        original_indices = list(range(n_features))
        K = []

        # Normalise the data ONCE at the beginning
        current_X = (X_orig - np.mean(X_orig, axis=0)) / np.std(X_orig, axis=0)
        # Compute the covariance matrix ONCE from the initial data
        current_cov = np.cov(current_X, rowvar=False, bias=True)

        for _ in range(n_features - 1):
            # Find the root using the current data and pre-computed covariance matrix
            # U is a list of indices relative to the current data matrix, e.g., [0, 1, 2, 3]
            U_current = list(range(current_X.shape[1]))
            m_in_current = self.search_causal_order_simplified(current_X, current_cov, U_current)

            # Map the index from the current space back to the original index
            original_idx_of_m = original_indices.pop(m_in_current)
            K.append(original_idx_of_m)

            # Update the data and covariance matrices for the next iteration
            # using the efficient formulas from the paper.
            current_X, current_cov = self.update_state(current_X, current_cov, m_in_current)

        # Append the final remaining variable
        if original_indices:
            K.append(original_indices[0])

        return K


if __name__ == '__main__':
    def get_matrix() -> pd.DataFrame:
        np.random.seed(42)
        x3 = np.random.uniform(size=1000)
        x0 = 3.0 * x3 + np.random.uniform(size=1000)
        x2 = 6.0 * x3 + np.random.uniform(size=1000)
        x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=1000)
        x5 = 4.0 * x0 + np.random.uniform(size=1000)
        x4 = 380 * x0 - 1.0 * x2 + np.random.uniform(size=1000)
        data = np.array([x0, x1, x2, x3, x4, x5]).T
        df = pd.DataFrame(data, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
        return df


    import time

    print("Running DirectLiNGAM with Mathematical Simplifications...")
    algorithm = DirectLingamCausalOrderAlgorithmLookup()
    start_time = time.time()
    causal_order = algorithm.run(get_matrix())
    end_time = time.time()

    print(f"Causal Order: {causal_order}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")