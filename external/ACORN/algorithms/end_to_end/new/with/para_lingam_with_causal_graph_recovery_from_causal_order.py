from algorithms.causal_order.new.para_lingam_causal_order_algorithm import ParaLingamCausalOrderAlgorithm
from algorithms.end_to_end.new.generic_algorithm_with_causal_graph_recovery_from_causal_order_ import \
    GenericAlgorithWithCausalGraphRecoveryFromCausalOrder


class ParaLingamWithCausalGraphRecoveryFromCausalOrder(GenericAlgorithWithCausalGraphRecoveryFromCausalOrder):
    def __init__(self):
        super().__init__(ParaLingamCausalOrderAlgorithm())