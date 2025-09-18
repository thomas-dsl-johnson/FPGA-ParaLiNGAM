class CausalOrderResult:
    """
    CausalOrderResult encapsulates the result of finding the causal order from a dataset
    """
    def __init__(self, causal_order: list[int], time_taken: float = None, algorithm_type: str="", algorithm_name: str = "", target_file: str = ""):
        self.causal_order = causal_order
        self.algorithm_type = algorithm_type
        self.algorithm_name = algorithm_name
        self.target_file = target_file
        self.time_taken = time_taken

    def __str__(self) -> str:
        sb = "Algorithm type: " + self.algorithm_type
        sb += "\nAlgorithm name: " + self.algorithm_name
        sb += "\nTarget file: " + self.target_file
        sb += "\nCausal order: " + str([int(x) for x in self.causal_order])
        sb += "\nTime taken: " + str(round(self.time_taken,2)) + " seconds"
        return sb.strip()