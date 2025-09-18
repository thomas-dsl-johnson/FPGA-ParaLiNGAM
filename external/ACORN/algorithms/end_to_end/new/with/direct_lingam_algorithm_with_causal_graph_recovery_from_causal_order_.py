from algorithms.causal_order.original.direct_lingam_causal_order_algorithm import DirectLingamCausalOrderAlgorithm
from algorithms.end_to_end.new.generic_algorithm_with_causal_graph_recovery_from_causal_order_ import \
    GenericAlgorithWithCausalGraphRecoveryFromCausalOrder


class  DirectLingamAlgorithmWithCausalGraphRecoveryFromCausalOrder(GenericAlgorithWithCausalGraphRecoveryFromCausalOrder):
    def __init__(self):
        super().__init__(DirectLingamCausalOrderAlgorithm())
