from algorithms.causal_order.new.direct_lingam_causal_order_algorithm_no_updates import \
    DirectLingamCausalOrderAlgorithmNoUpdates
from algorithms.end_to_end.new.generic_algorithm_with_causal_graph_recovery_from_causal_order_ import \
    GenericAlgorithWithCausalGraphRecoveryFromCausalOrder


class NoUpdatesAlgorithmWithCausalGraphRecoveryFromCausalOrder(GenericAlgorithWithCausalGraphRecoveryFromCausalOrder):
    def __init__(self):
        super().__init__(DirectLingamCausalOrderAlgorithmNoUpdates())
