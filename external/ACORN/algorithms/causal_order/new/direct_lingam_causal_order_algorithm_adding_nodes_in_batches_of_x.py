"""
Add nodes in batches of x : x is an integer
"""
BATCH_SIZE = 10
import math
import pandas as pd
import numpy as np
from algorithms.causal_order.generic_causal_order_algorithm import GenericCausalOrderAlgorithm


class DirectLingamCausalOrderAlgorithmAddingNodesInBatchesOfX(GenericCausalOrderAlgorithm):
    """
    Runs the DirectLiNGAM algorithm to generate the causal order:
    1.  Given a p-dimensional random vector x, a set of its variable subscripts U and a p × n data
        matrix of the random vector as X, initialize an ordered list of variables K := /0 and m := 1.
    2. Repeat until p−1 subscripts are appended to K:
        (a) Perform least squares regressions of xi on x j for all i ∈ U minus K (i 6 = j) and compute the
            residual vectors r( j) and the residual data matrix R( j) from the data matrix X for all
            j ∈ U minus K. Find a variable xm that is most independent of its residuals:
            xm = arg min
            j∈U minus K Tkernel (x j;U minus K),
            where Tkernel is the independence measure defined in Equation (7).
        (b) Append m to the end of K.
        (c) Let x := r(m), X := R(m).
    3.  Append the remaining variable to the end of K.
    """

    def run(self, df: pd.DataFrame) -> list[int]:
        """
        Run the Causal Order Algorithm

        Parameters
        ----------
        df : pd.DataFrame,
             The training IT_monitoring, with shape (n_samples, n_features) where
             - n_samples is the number of samples
             - n_features is the number of features

        Returns
        ----------
        causal_order : CausalOrder,
            The causal order. A CausalOrder is a list[int] where
            each integer represents the index of a feature in the training IT_monitoring.
        """
        return self.get_causal_order_using_direct_lingam(df)

    def __str__(self) -> str:
        return "DirectLingamAlgorithmAddingNodesInBatchesOfX"

    @staticmethod
    def residual(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """
        Compute the residual of xi after linear regression on xj.

        This function calculates the residuals from a simple linear regression of `xi` on `xj`.
        It subtracts the linear prediction of `xi` based on `xj` from the actual `xi` values.

        Parameters
        ----------
        xi : list of float
            The dependent variable, a sequence of numerical observations.
        xj : list of float
            The independent variable, a sequence of numerical observations used to predict `xi`.

        Returns
        -------
        residuals : np.ndarray
            A NumPy array of residual values, i.e., the difference between `xi` and its linear
            projection onto `xj`. The array has the same shape as the input lists.
        """
        return xi - (np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)) * xj

    @staticmethod
    def entropy(u: np.ndarray) -> float:
        """
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
        return (1 + np.log(2 * np.pi)) / 2 - k1 * (
                np.mean(np.log(np.cosh(u))) - gamma) ** 2 - k2 * (np.mean(u * np.exp((-(u ** 2)) / 2))) ** 2

    def diff_mutual_info(self, xi_std: np.ndarray, xj_std: np.ndarray, ri_j: np.ndarray, rj_i: np.ndarray) -> float:
        """
        Calculate the difference in mutual information between two variables.

        This function estimates the difference in mutual information between two standardized variables
        `xi_std` and `xj_std`, and their respective residuals `ri_j` and `rj_i`.

        Parameters
        ----------
        xi_std : np.ndarray
            Standardized version of variable xi.
        xj_std : np.ndarray
            Standardized version of variable xj.
        ri_j : np.ndarray
            Residuals of xi regressed on xj.
        rj_i : np.ndarray
            Residuals of xj regressed on xi.

        Returns
        -------
        diff : float
            The difference in estimated mutual information:
            (entropy(xj_std) + entropy(ri_j)) - (entropy(xi_std) + entropy(rj_i)).
        """
        return (self.entropy(xj_std) + self.entropy(ri_j / np.std(ri_j))) - (
                self.entropy(xi_std) + self.entropy(rj_i / np.std(rj_i))
        )

    def search_causal_order(self, X: np.ndarray, U: list[int], batch_size=None) -> list[int]:
        """
        Search for the next variable in the causal ordering.

        This function selects the most likely "root cause" variable from a set of candidates `U`,
        based on the principle that causes are more independent of their residuals after removing
        the effect of other variables.

        Parameters
        ----------
        X : np.ndarray
            The standardised data matrix of shape (n_samples, n_features),
            where each column represents a variable.
        U : list[int]
            A list of indices representing the remaining candidate variables
            from which the next root cause should be selected.

        Returns
        -------
        xm : int
            The index of the selected variable that is estimated to be the most upstream (least caused)
        """
        # Uc: In this implementation, search_candidate simply returns U, meaning all remaining variables are candidates.
        # If prior knowledge were available (e.g., "variable A cannot cause variable B"),
        # this function would prune the candidate set.
        Uc = U
        # if there is only one candidate feature - we are done
        if len(Uc) <= batch_size:
            # If fewer or equal than batch_size, just return them all
            return list(Uc)
        # M_list: Stores the computed M values for each candidate variable.
        M_list = []
        for i in Uc:
            M = 0
            for j in Uc:
                if i != j:
                    # We compare two unique nodes i,j
                    # xi_std, xj_std: Standardised versions of variables i and j.
                    #                 Standardisation helps in comparing magnitudes across different variables.
                    # ri_j: This calculates the residual of xi_std when regressed on xj_std.
                    #       In essence, it removes the linear effect of xj_std from xi_std.
                    #       If xj causes xi,
                    #       then the residual ri_j should be more "independent" or non-Gaussian than xi itself.
                    # rj_i: Similarly, the residual of xj_std when regressed on xi_std.

                    #  Then compute the related independence measure between two scenarios:
                    #   The independence of xj and the residual ri_j (where xi is regressed on xj)
                    #   The independence of xi and the residual rj_i (where xj is regressed on xi).
                    #   The entropy function is used as a proxy for non-Gaussianity or independence.
                    #   The idea is that if j is a cause of i,
                    #   then ri_j (the effect i with the cause j removed)
                    #   should be more independent (or have higher entropy) than rj_i.
                    #   The np.min([0, ...]) ** 2 part penalizes positive values,
                    #   effectively minimizing a quantity that approaches zero when one variable is a true cause.
                    xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                    xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
                    ri_j = (self.residual(xi_std, xj_std))
                    rj_i = (self.residual(xj_std, xi_std))
                    M += np.min([0, self.diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
            M_list.append(-1.0 * M)

        #  xm has the maximum M value
        #  This corresponds to the minimum of the negative sum,
        #  effectively finding the variable that is "most independent" of its residuals
        #  It is selected as the next variable in the causal order.
        #  This xm is considered to be a "root" cause among the remaining variables.

        ##############
        ##############
        top_indices = np.argsort(M_list)[-batch_size:]  # largest batch_size M's
        top_nodes = [Uc[idx] for idx in reversed(top_indices)]  # reverse for descending order

        return top_nodes
        ##############
        ##############


    def get_causal_order_using_direct_lingam(self, df: pd.DataFrame) -> list[int]:
        """
        Estimate the causal order of variables using the DirectLiNGAM algorithm.

        This function implements the DirectLiNGAM (Linear Non-Gaussian Acyclic Model) algorithm to
        discover a causal ordering among variables in a multivariate dataset. It iteratively identifies
        the most independent variable (assumed to be a root cause), removes its linear effects from
        the others, and continues the process with the residuals.

        Parameters
        ----------
        df : np.ndarray
            A data matrix of shape (n_samples, n_features), where each column is a variable and each
            row is an observation. Variables should be linearly related but non-Gaussian.

        Returns
        -------
        K : list of int
            A list of variable indices representing the estimated causal order.
            The first element is the most upstream (least caused) variable, and the last is the most downstream.
        """
        # Step 1: initialise
        # X: A p × n data matrix of the random vector,
        # where p is the number of dimensions (variables) and n is the number of samples
        # U: A set of variable subscripts, initially containing all variable indices (from 0 to p-1).
        # K: An empty ordered list, which will store the discovered causal order
        # X_: A copy of the original data matrix X.
        # This copy will be modified throughout the algorithm by computing residuals.
        n_features = df.shape[1]
        U = np.arange(n_features)
        K = []
        X_ = np.copy(df)

        batch_size = BATCH_SIZE

        while len(U) > 0:
            # Step 2(a): Find the batch_size Most Independent Variables m
            next_nodes = self.search_causal_order(X_, U, batch_size=batch_size)
            # Step 2(b): Append newly found causal variable m to ordered list K
            # Append all selected nodes to K
            K.extend(next_nodes)

            # Step 2(c): Update X and U:
            # For each node in U but not in next_nodes, remove linear effect of these next_nodes
            # For all other variables i still in U (not yet ordered),
            # their values in X_ are updated by regressing them on m.
            # This means the linear effect of m is removed from i.
            # If xm is a cause, its influence should be removed from its effects
            # to discover further causal relationships among the remaining variables.
            # This effectively transforms x into r(m) and X into R(m) as described in the algorithm.
            remaining = [i for i in U if i not in next_nodes]
            for node in next_nodes:
                for i in remaining:
                    X_[:, i] = self.residual(X_[:, i], X_[:, node])

            # Step 2(d) m is removed from the set of unordered variables U.
            # Remove these nodes from U
            U = np.array([i for i in U if i not in next_nodes])
        return K


if __name__ == '__main__':
    def get_matrix() -> pd.DataFrame:
        """
        Return a valid input matrix with type pandas Dataframe

        Returns
        ----------
        df : pd.Dataframe
        """
        x3 = np.random.uniform(size=1000)
        x0 = 3.0 * x3 + np.random.uniform(size=1000)
        x2 = 6.0 * x3 + np.random.uniform(size=1000)
        x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=1000)
        x5 = 4.0 * x0 + np.random.uniform(size=1000)
        x4 = 380 * x0 - 1.0 * x2 + np.random.uniform(size=1000)
        data = np.array([x0, x1, x2, x3, x4, x5]).T
        df = pd.DataFrame(data, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
        return df


    algorithm = DirectLingamCausalOrderAlgorithmAddingNodesInBatchesOfX()
    print(algorithm.get_causal_order_using_direct_lingam(get_matrix()))
