from typing import Tuple, Any, List

import numpy as np
from algorithms.causal_order.generic_causal_order_algorithm import GenericCausalOrderAlgorithm, CausalOrder
from algorithms.end_to_end.end_to_end_result import EndToEndResult
from algorithms.end_to_end.generic_end_to_end_algorithm import GenericEndToEndAlgorithm
from external.recover_causal_graph_from_causal_order.utils.estimate_adjacency_matrix import estimate_adjacency_matrix


class GenericAlgorithWithCausalGraphRecoveryFromCausalOrder(GenericEndToEndAlgorithm):
    """
    This class is initialised with a causal order algorithm. i.e. an algorithm that produces a causal order.
    This algorithm is composed with the Causal Graph Recovery Algorithm to provide a full end-to-end algorithm.

    A picture:
    1. Causal Order Algorithm: Input -> Causal Order
    2. Causal Graph Recovery Algorithm: Causal Order -> Model

    See here for information on the algorithm:
    https://github.com/jultrishyyy/Recover-Causal-Graph-from-Causal-Order
    """
    def __init__(self, causal_order_algorithm: GenericCausalOrderAlgorithm):
        self.causal_order_algorithm = causal_order_algorithm

    def format_result(self, result: tuple[np.ndarray, list[int]], time_taken: float) -> EndToEndResult:
        return EndToEndResult.from_matrix(result, time_taken, self._algorithm_type(), self.__str__(), self.target_file)

    def run(self, df) -> tuple[np.ndarray, list[int]]:
        res = self.causal_order_algorithm.run(df)
        causal_order = res

        # DATA_PATH = args.data_path
        # OUTPUT_PATH = args.output_path
        # os.makedirs(OUTPUT_PATH, exist_ok=True)

        # label_filename = os.path.join(DATA_PATH, 'summary_matrix.npy')
        # # Format variables to match code in recover_causal_graph_from_causal_order
        # output_metrics_filename = os.path.join(OUTPUT_PATH, 'metrics.txt')
        # output_graph_filename = os.path.join(OUTPUT_PATH, 'causal_graph.png')

        X = df.values

        # Load causal order from file
        causal_order = [int(x) for x in causal_order]

        # Estimate adjacency matrix
        estimated_summary_matrix_continuous = estimate_adjacency_matrix(causal_order, X)
        return (estimated_summary_matrix_continuous, causal_order)

        # # Load label summary matrix
        # label_summary_matrix = np.load(label_filename)
        # # Prune estimated summary matrix with best F1 threshold
        # estimated_summary_matrix = prune_summary_matrix_with_best_f1_threshold(estimated_summary_matrix_continuous,label_summary_matrix)
        # print("\nEstimated summary matrix:")
        # print(estimated_summary_matrix)
        # # Optional: save the evaluation results and metrics
        # save_results_and_metrics(
        #     label_summary_matrix,
        #     estimated_summary_matrix,
        #     estimated_summary_matrix_continuous,
        #     order=causal_order,
        #     filename=output_metrics_filename
        # )
        # # Plot and save the causal graph
        # plot_summary_causal_graph(estimated_summary_matrix, filename=output_graph_filename)
        # # Optionally, plot the ground truth causal graph
        # # plot_summary_causal_graph(label_summary_matrix, filename=output_graph_filename.replace('.png', '_label.png'))

    def __str__(self):
        return self.causal_order_algorithm.__str__() + "_followed_by_CausalGraphRecoveryFromCausalOrder"