import pandas as pd
import numpy as np
from algorithms.causal_order.generic_causal_order_algorithm import GenericCausalOrderAlgorithm


class DirectLingamCausalOrderAlgorithmThreshold(GenericCausalOrderAlgorithm):
    """
    Implements the DirectLiNGAM algorithm but incorporates the 'Threshold Mechanism'
    from the ParaLiNGAM paper to accelerate the root-finding step[cite: 10].

    The core idea is to set an upper limit (a threshold) on the dependency score.
    If a variable's score exceeds this limit during the comparison phase, we assume
    it cannot be the root cause and terminate further comparisons for it, saving computation[cite: 311, 315].
    If all variables exceed the threshold, it is increased, and the process repeats[cite: 316].
    """

    def run(self, df: pd.DataFrame) -> list[int]:
        """
        Run the Causal Order Algorithm.
        (This method is identical to the original implementation's entry point).
        """
        return self.get_causal_order_using_direct_lingam(df)

    def __str__(self) -> str:
        return "ThresholdLingamAlgorithm"

    # The static helper methods (residual, entropy, diff_mutual_info) are identical to original
    @staticmethod
    def residual(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        return xi - (np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)) * xj

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

    def search_causal_order(self, X: np.ndarray, U: list[int]) -> int:
        """
        Search for the next root variable using the threshold mechanism.
        """
        Uc = U
        if len(Uc) == 1:
            return Uc[0]

        # --- Threshold Mechanism Initialisation ---
        # Start with a very small threshold[cite: 314].
        gamma = 1e-5
        # Set a factor to increase the threshold if needed[cite: 340].
        threshold_increase_factor = 10.0

        while True:
            scores = {}
            # Track which variables completed comparisons without exceeding the threshold.
            completed_successfully = {}

            # This loop iterates through all candidate variables to calculate their scores.
            for i in Uc:
                M = 0
                breached_threshold = False
                # Inner loop compares variable 'i' against all others.
                for j in U:
                    if i == j:
                        continue

                    # --- Threshold Check ---
                    # If the score 'M' for variable 'i' is already above the threshold,
                    # break the inner loop to save computations[cite: 315].
                    if M > gamma:
                        breached_threshold = True
                        break

                    xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                    xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
                    ri_j = (self.residual(xi_std, xj_std))
                    rj_i = (self.residual(xj_std, xi_std))
                    M += np.min([0, self.diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2

                scores[i] = M
                if not breached_threshold:
                    # This variable is a potential root cause for the current threshold.
                    completed_successfully[i] = M

            # --- Check for Termination or Threshold Update ---
            # The iteration is over if at least one variable finished its comparisons
            # without its score exceeding the threshold[cite: 312, 317].
            if completed_successfully:
                # Find the variable with the minimum score among the successful ones.
                # This is our root.
                root_variable = min(completed_successfully, key=completed_successfully.get)
                return root_variable
            else:
                # If all variables breached the threshold, we must increase it and
                # re-run the comparisons[cite: 316, 338].
                gamma *= threshold_increase_factor

    def get_causal_order_using_direct_lingam(self, df: pd.DataFrame) -> list[int]:
        """
        This is the main driver loop from the original file, which remains unchanged.
        It repeatedly calls the (now modified) `search_causal_order` method.
        """
        n_features = df.shape[1]
        U = np.arange(n_features)
        K = []
        X_ = np.copy(df)

        for _ in range(n_features):
            m = self.search_causal_order(X_, U)
            K.append(m)

            # This check is to avoid error on the last iteration
            if len(U) == 1:
                break

            for i in U:
                if i != m:
                    X_[:, i] = self.residual(X_[:, i], X_[:, m])
            U = U[U != m]
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

    print("Running DirectLiNGAM with Threshold Mechanism...")
    algorithm = DirectLingamCausalOrderAlgorithmThreshold()
    start_time_thresh = time.time()
    causal_order_thresh = algorithm.run(get_matrix())
    end_time_thresh = time.time()

    print(f"Causal Order: {causal_order_thresh}")
    print(f"Execution Time: {end_time_thresh - start_time_thresh:.4f} seconds")