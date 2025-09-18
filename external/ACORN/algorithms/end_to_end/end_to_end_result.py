import lingam
import numpy as np
from algorithms.causal_order.causal_order_result import CausalOrderResult
from algorithms.causal_order.generic_causal_order_algorithm import CausalOrder


class EndToEndResult:
    """
    EndtoEndLingamResult encapsulates the result of finding the causal order and model from a dataset
    """

    def __init__(self, causal_order_result: CausalOrderResult, summary_matrix: np.ndarray,
                 model: lingam.DirectLiNGAM = None):
        self.causal_order_result = causal_order_result
        self.summary_matrix = summary_matrix
        self.model: lingam.DirectLiNGAM = model

    @classmethod
    def from_model(self, model: lingam.DirectLiNGAM, time_taken: float, algorithm_type: str, algorithm_name: str, target: str) -> 'EndToEndResult':
        causal_order_result = CausalOrderResult(model.causal_order_, time_taken, algorithm_type, algorithm_name, target)
        return self(causal_order_result, model.adjacency_matrix_, model=model)

    @classmethod
    def from_matrix(self,  result: tuple[np.ndarray, CausalOrder], time_taken: float, algorithm_type: str, algorithm_name: str, target: str) -> 'EndToEndResult':
        estimated_summary_matrix_continuous, causal_order = result
        causal_order_result = CausalOrderResult(causal_order, time_taken, algorithm_type, algorithm_name, target)
        return self(causal_order_result, estimated_summary_matrix_continuous,)

    def __str__(self) -> str:
        sb = ""
        # Summary Matrix
        if self.model != None:
            sb += f"\nSummary matrix: \n{self.model.adjacency_matrix_}"
        elif self.summary_matrix is not None:
            sb += f"\nSummary matrix: \n{self.summary_matrix}"
        return str(self.causal_order_result) + sb
