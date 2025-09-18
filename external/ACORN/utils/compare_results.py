"""
A set of functions that compare two results
"""
import os
import numpy as np
from pathlib import Path

from algorithms.causal_order.causal_order_result import CausalOrderResult
from algorithms.causal_order.generic_causal_order_algorithm import CausalOrder
from algorithms.end_to_end.end_to_end_result import EndToEndResult
from algorithms.end_to_end.original.direct_lingam_end_to_end_algorithm import DirectLingamEndToEndAlgorithm
from utils import storage

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_ground_truth_causal_order(filepath: str) -> (CausalOrder, bool):
    """
    Retrieve the ground truth causal order for a given dataset.

    Depending on the dataset's directory structure, the function either:
    - Loads the causal order from a `causal_order.txt` file if ground truth is available.
    - Loads a proxy "ground truth" causal order from the stored `DirectLiNGAM` end-to-end result
      if ground truth is not available.

    Parameters
    ----------
    filepath : str
        Path to the dataset file. The path must contain either
        `"ground_truth_available"` or `"ground_truth_not_available"` in its folder structure.

    Returns
    -------
    causal_order : CausalOrder
        A list of feature indices representing the true (or proxy) causal order,
        where earlier indices are causes of later indices.
    has_ground_truth : bool
        True if the causal order is an actual ground truth, False if it is derived
        from `DirectLiNGAM` results.

    Raises
    ------
    FileNotFoundError
        If the expected `causal_order.txt` or the `DirectLiNGAM` result file cannot be found.
    Exception
        If the `filepath` does not contain a recognized ground truth status directory.

    Notes
    -----
    - Ground truth datasets are expected to store their order in the same folder
      as the dataset file under the name `causal_order.txt`.
    - The `causal_order.txt` file should contain a Python-style list of integers, e.g. `[0, 1, 2]`.
    - For datasets without ground truth, the function looks for:
      `"end_to_end/<DirectLiNGAMAlgorithmName>/<dataset_dir>/<dataset_name>.pkl"`.
    """
    data_path = Path(filepath)
    if "ground_truth_available" in data_path.parts:
        # Ground truth exists
        # Retrieve ground truth from same folder as dataset
        causal_order_path = data_path.parent / "causal_order.txt"
        if not causal_order_path.exists():
            raise FileNotFoundError(f"No causal_order.txt found at {causal_order_path}")
        with open(causal_order_path, "r") as f:
            line = f.readline().strip("[]").split(",")
            causal_order_list = [int(x) for x in line]
        return causal_order_list, True
    elif "ground_truth_not_available" in data_path.parts:
        # No ground truth
        # Retrieve 'ground truth' from DirLingam EndtoEnd result
        result_path = "end_to_end" + "/" + DirectLingamEndToEndAlgorithm().__str__() + "/" + os.path.basename(
            os.path.dirname(filepath)) + "/" + os.path.splitext(os.path.basename(filepath))[0] + ".pkl"
        if storage.exists(result_path):
            res: EndToEndResult = storage.load(result_path)
            return res.causal_order_result.causal_order, False
        else:
            raise FileNotFoundError(f"No file found at {result_path}")
    else:
        raise Exception(f"Unknown result type {data_path}")

def get_ground_truth_summary_matrix(filepath: str) -> (CausalOrder, bool):
    """
    Retrieve the ground truth summary matrix for a given dataset.

    Depending on the dataset's directory structure, the function either:
    - Loads the ground truth matrix from a `summary_matrix.npy` file if ground truth is available.
    - Loads a proxy "ground truth" from the stored `DirectLiNGAM` end-to-end result
      if ground truth is not available.

    Parameters
    ----------
    filepath : str
        Path to the dataset file. The path must contain either
        `"ground_truth_available"` or `"ground_truth_not_available"` in its folder structure.

    Returns
    -------
    summary_matrix : np.ndarray
        The binary adjacency matrix of shape (n_features, n_features) representing
        the ground truth or proxy causal structure.
    has_ground_truth : bool
        True if the matrix is an actual ground truth, False if it is derived from
        `DirectLiNGAM` results.

    Raises
    ------
    FileNotFoundError
        If the expected `summary_matrix.npy` or the `DirectLiNGAM` result file cannot be found.
    Exception
        If the `filepath` does not contain a recognized ground truth status directory.

    Notes
    -----
    - Ground truth datasets are expected to store their matrix in the same folder
      as the dataset file under the name `summary_matrix.npy`.
    - For datasets without ground truth, the function looks for:
      `"end_to_end/<DirectLiNGAMAlgorithmName>/<dataset_dir>/<dataset_name>.pkl"`.
    """
    data_path = Path(filepath)
    if "ground_truth_available" in data_path.parts:
        # Ground truth exists
        # Retrieve ground truth from same folder as dataset
        summary_matrix_path = data_path.parent / "summary_matrix.npy"
        if not summary_matrix_path.exists():
            raise FileNotFoundError(f"No summary_matrix.npy found at {summary_matrix_path}")
        summary_matrix = np.load(summary_matrix_path)
        return summary_matrix, True
    elif "ground_truth_not_available" in data_path.parts:
        # No ground truth
        # Retrieve 'ground truth' from DirLingam EndtoEnd result
        result_path = "end_to_end" + "/" + DirectLingamEndToEndAlgorithm().__str__() + "/" + os.path.basename(
            os.path.dirname(filepath)) + "/" + os.path.splitext(os.path.basename(filepath))[0] + ".pkl"
        if storage.exists(result_path):
            res: EndToEndResult = storage.load(result_path)
            return res.summary_matrix, False
        else:
            raise FileNotFoundError(f"No file found at {result_path}")
    else:
        raise Exception(f"Unknown result type {data_path}")


def get_result_file_causal_order(dataset_path: str, algorithm_name: str) -> CausalOrder:
    """
    Load the predicted causal order from a stored result file.

    This function constructs the expected file path to the stored causal order result
    for a given dataset and algorithm, validates that the dataset path is under the
    project's `data` directory, and returns the saved causal order.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset file. Must be located under `PROJECT_ROOT / "data"`.
    algorithm_name : str
        Name of the algorithm whose causal order result should be loaded.

    Returns
    -------
    CausalOrder
        A list of feature indices representing the predicted causal order,
        where earlier indices are predicted causes of later indices.

    Raises
    ------
    ValueError
        If `dataset_path` is not located under the expected `data` directory.

    Notes
    -----
    - The causal order results are stored in:
      `PROJECT_ROOT / "results" / "causal_order" / algorithm_name / <subdir> / <dataset_name>.pkl`
    - The stored file is expected to be a `CausalOrderResult` object with a `.causal_order` attribute.
    """
    dataset_path = Path(dataset_path)
    dataset_filename = dataset_path.stem + ".pkl"

    # Extract the relative path from the dataset root
    try:
        relative_path = dataset_path.relative_to(PROJECT_ROOT / "data")
    except ValueError:
        raise ValueError(f"Dataset path {dataset_path} is not under the expected 'data' directory.")

    # Drop the first component (e.g., "ground_truth_available")
    parts = list(relative_path.parts)[1:-1]  # Skip file name and root folder
    subdir = Path(*parts) if parts else Path()

    # Full result path
    result_file_path = PROJECT_ROOT / "results" / "causal_order" / algorithm_name / subdir / dataset_filename
    causal_order_result : CausalOrderResult = storage.load(str(result_file_path))
    return causal_order_result.causal_order

def get_result_file_summary_matrix(dataset_path: str, algorithm_name: str) -> np.ndarray:
    """
    Load the summary matrix from an end-to-end result file for a given dataset and algorithm.

    This function constructs the file path to the stored end-to-end result,
    verifies that the dataset path is under the expected `data` directory,
    and loads the corresponding summary matrix.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset file. Must be located under `PROJECT_ROOT / "data"`.
    algorithm_name : str
        Name of the algorithm used to produce the results.

    Returns
    -------
    np.ndarray
        The summary matrix from the loaded end-to-end result file.

    Raises
    ------
    ValueError
        If the provided `dataset_path` is not under the expected `data` directory.

    Notes
    -----
    - The dataset file name is assumed to be converted to a `.pkl` file for results.
    - Results are stored in:
      `PROJECT_ROOT / "results" / "end_to_end" / algorithm_name / <subdir> / <dataset_name>.pkl`
    - The `<subdir>` is derived from the dataset path structure by dropping the first folder
      and the file name.
    """
    dataset_path = Path(dataset_path)
    dataset_filename = dataset_path.stem + ".pkl"

    # Extract the relative path from the dataset root
    try:
        relative_path = dataset_path.relative_to(PROJECT_ROOT / "data")
    except ValueError:
        raise ValueError(f"Dataset path {dataset_path} is not under the expected 'data' directory.")

    # Drop the first component (e.g., "ground_truth_available")
    parts = list(relative_path.parts)[1:-1]  # Skip file name and root folder
    subdir = Path(*parts) if parts else Path() # Maybe change file structure later
    # Full result path
    result_file_path = PROJECT_ROOT / "results" / "end_to_end" / algorithm_name / subdir / dataset_filename
    end_to_end_result : EndToEndResult = storage.load(str(result_file_path))
    return end_to_end_result.summary_matrix

def count_inverted_pairs(discovered_causal_order: CausalOrder, ground_truth_matrix: np.ndarray) -> float:
    """
    Count the proportion of inverted causal pairs in a discovered causal order.

    This function compares a predicted causal ordering with the ground truth causal
    adjacency matrix and measures how often the predicted order violates the
    true cause-effect direction.

    Parameters
    ----------
    discovered_causal_order : CausalOrder
        A list of feature indices representing the predicted causal order,
        where earlier elements are predicted to be causes of later ones.
    ground_truth_matrix : np.ndarray
        The ground truth adjacency matrix of shape (n_features, n_features),
        where a non-zero entry at (i, j) means feature j causes feature i.

    Returns
    -------
    float
        The fraction of inverted causal pairs, calculated as:
        (number of incorrectly ordered pairs) / (total number of possible ordered pairs).

    Notes
    -----
    - A pair (j → i) is considered inverted if `j` appears after `i`
      in `discovered_causal_order`.
    - The denominator counts all possible ordered feature pairs,
      regardless of whether a causal relation exists.
    """
    def correct_in_causal_order(order, j, i):
        return order.index(j) < order.index(i)

    total_number_of_pairs = len(discovered_causal_order) * (len(discovered_causal_order) - 1)
    inverted_pairs = 0
    A = ground_truth_matrix
    n = len(A)
    for i in range(0, n):
        for j in range(0, n):
            x = A[i, j]
            if x != 0:  # j -> i
                if not correct_in_causal_order(discovered_causal_order, j, i):
                    inverted_pairs += 1
    return inverted_pairs / total_number_of_pairs



def save_results_and_metrics(label_summary_matrix, estimated_summary_matrix, estimated_summary_matrix_continuous=None,
                             lags=None, order=None, filename="results.txt", additional_info=None):
    """
    From Zhao Tong's Repository: https://github.com/jultrishyyy/Recover-Causal-Graph-from-Causal-Order/tree/main

    Save evaluation metrics, causal order analysis, and summary matrices to a text file.

    This function compares a ground-truth binary adjacency matrix with an estimated one,
    calculates performance metrics (precision, recall, F1), optionally evaluates causal order correctness,
    and writes all results to a specified file.

    Parameters
    ----------
    label_summary_matrix : np.ndarray
        Binary ground-truth adjacency matrix of shape (n_features, n_features).
        1 indicates a causal edge exists, 0 indicates no edge.
    estimated_summary_matrix : np.ndarray
        Binary predicted adjacency matrix of the same shape as `label_summary_matrix`,
        representing pruned predictions after causal discovery.
    estimated_summary_matrix_continuous : np.ndarray, optional
        Continuous-valued adjacency matrix (e.g., maximum estimated causal effect before pruning).
        Same shape as `label_summary_matrix`.
    lags : list[int], optional
    order : list[int], optional
        The predicted causal order of features, where each integer is a feature index.
    filename : str, default="results.txt"
        Path to the output file where results will be saved.
    additional_info : list[str], optional
        Additional textual information to be written at the start of the file.

    Returns
    -------
    None
        The results are written directly to the specified file.

    Notes
    -----
    - Precision, recall, and F1 score are computed based on binary matrices.
    - If `order` is provided, the function also counts wrongly ordered causal pairs.
    - The estimated summary matrix is saved in CSV format within the text file.
    """
    with open(filename, 'w') as f:
        # --- Write scalar values ---
        if additional_info is not None:
            f.write("--- ADDITIONAL INFO ---\n\n")
            for info in additional_info:
                f.write(f"{info}\n")

        f.write("\n\n--- METRICS ---\n\n")

        if lags is not None:
            f.write(f"Best Lags: {lags}\n")

        num_edges_ground_truth = np.sum(label_summary_matrix)
        f.write(f"Number of edges in ground truth (label summary matrix): {num_edges_ground_truth}\n")

        # True Positives: in both labels and pruned prediction
        true_positives_matrix = (label_summary_matrix == 1) & (estimated_summary_matrix == 1)
        num_correctly_predicted = np.sum(true_positives_matrix)
        f.write(f"Number of correctly predicted edges (True Positives): {num_correctly_predicted}\n")

        # True Negatives: not in both labels and pruned prediction
        true_negatives_matrix = (label_summary_matrix == 0) & (estimated_summary_matrix == 0)
        num_correct_nonedges = np.sum(true_negatives_matrix)
        f.write(f"Number of correct non-edges (True Negatives): {num_correct_nonedges}\n")

        # False Positives: in pruned prediction but not in labels
        false_positives_matrix = (label_summary_matrix == 0) & (estimated_summary_matrix == 1)
        num_incorrectly_predicted = np.sum(false_positives_matrix)
        f.write(f"Number of incorrectly predicted edges (False Positives): {num_incorrectly_predicted}\n")

        # False Negatives: in labels but not in pruned prediction
        false_negatives_matrix = (label_summary_matrix == 1) & (estimated_summary_matrix == 0)
        num_missed_edges = np.sum(false_negatives_matrix)
        f.write(f"Number of correct edges not predicted (False Negatives): {num_missed_edges}\n")

        tp = num_correctly_predicted
        fp = num_incorrectly_predicted
        fn = num_missed_edges

        # Calculate Precision
        if (tp + fp) == 0:
            precision = 0.0  # Avoid division by zero if no positive predictions were made
        else:
            precision = tp / (tp + fp)

        # Calculate Recall
        if (tp + fn) == 0:
            recall = 0.0  # Avoid division by zero if no actual positives exist (or none were predicted)
        else:
            recall = tp / (tp + fn)

        # Calculate F1 Score
        if (precision + recall) == 0:
            f1_score = 0.0  # Avoid division by zero if both precision and recall are zero
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n")

        if order is not None:
            f.write("\n--- ORDER ANALYSIS---\n\n")
            f.write(f"Predicted Causal Order: {order}\n")

            num_wrongly_ordered_edges = 0
            wrong_pairs = []

            if num_correctly_predicted > 0:
                # Get indices of true positive edges
                # np.where returns a tuple of arrays, one for each dimension
                tp_effect_indices, tp_cause_indices = np.where(label_summary_matrix == 1)
                # print(f"True Positive Effect Indices: {tp_effect_indices}")
                # print(f"True Positive Cause Indices: {tp_cause_indices}")

                for i in range(len(tp_effect_indices)):
                    effect_idx = tp_effect_indices[i]
                    cause_idx = tp_cause_indices[i]

                    # Check order: if cause's order is >= effect's order, it's wrong
                    if order.index(cause_idx) > order.index(effect_idx):
                        num_wrongly_ordered_edges += 1
                        wrong_pairs.append(f"    - Wrongly ordered: {cause_idx} -> {effect_idx} \n")

            f.write(f"Number of wrongly ordered cause-effect pairs: {num_wrongly_ordered_edges}\n")

            print_num = min(len(wrong_pairs), 5)
            f.write(f"{print_num} Wrongly ordered pairs in ground truth causal matrix:\n")
            if num_wrongly_ordered_edges > 0:
                for pair in wrong_pairs[:print_num]:
                    f.write(pair)

        f.write("\n\n" + "=" * 50 + "\n\n")

        # --- Write the summary_matrix_continuous ---
        f.write("--- ESTIMATED SUMMARY MATRIX ---\n")
        f.write("(Represents the estimated causal effect across all lags after pruning)\n\n")
        np.savetxt(f, estimated_summary_matrix, delimiter=',', fmt='%d')

        f.write("\n\n\n" + "=" * 50 + "\n\n")

        # # --- Write the summary_matrix_continuous ---
        # f.write("--- CONTINUOUS SUMMARY MATRIX ---\n")
        # f.write("(Represents the max estimated causal effect across all lags before pruning)\n\n")
        #
        # # 6. Use numpy.savetxt to write the array to the file handle 'f'
        # # 'fmt' controls the number format to keep it clean.
        # np.savetxt(f, estimated_summary_matrix_continuous, fmt='%.6f', delimiter=',')

    print(f"All results have been successfully saved to '{filename}'")

if __name__ == "__main__":
    dataset = "/Users/thomasjohnson/Desktop/UROP/ACORN/data/ground_truth_available/Causal_River/Flood/rivers_ts_flood_preprocessed.csv"
    ground_truth_causal_order, isRealTruth = get_ground_truth_causal_order(dataset)
    ground_truth_summary_matrix, _ = get_ground_truth_summary_matrix(dataset)
    print(ground_truth_causal_order, isRealTruth)
    print(ground_truth_summary_matrix)
    algorithm_name = "ParaLingamAlgorithm"
    if isRealTruth:
        algorithm_name += ("_followed_by_CausalGraphRecoveryFromCausalOrder")
        found_summary_matrix = get_result_file_summary_matrix(dataset, algorithm_name)
        save_results_and_metrics(ground_truth_summary_matrix, found_summary_matrix)
    else:
        found_causal_order = get_result_file_causal_order(dataset, algorithm_name)
        print("Ratio of inverted pairs: ", count_inverted_pairs(found_causal_order, ground_truth_summary_matrix))