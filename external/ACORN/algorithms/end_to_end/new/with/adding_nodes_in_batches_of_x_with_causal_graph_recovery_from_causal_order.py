from algorithms.causal_order.new.direct_lingam_causal_order_algorithm_adding_nodes_in_batches_of_x import \
    DirectLingamCausalOrderAlgorithmAddingNodesInBatchesOfX
from algorithms.end_to_end.new.generic_algorithm_with_causal_graph_recovery_from_causal_order_ import \
    GenericAlgorithWithCausalGraphRecoveryFromCausalOrder


class  AddingNodesInBatchesOfXWithCausalGraphRecoveryFromCausalOrder(GenericAlgorithWithCausalGraphRecoveryFromCausalOrder):
    def __init__(self):
        super().__init__(DirectLingamCausalOrderAlgorithmAddingNodesInBatchesOfX())
