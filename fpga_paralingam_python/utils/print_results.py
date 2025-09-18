"""
Find the results within the algorithms/ directory and prints a summary of each one
"""
import sys
from pathlib import Path
from typing import Any, List

from algorithms.causal_order.causal_order_result import CausalOrderResult
from algorithms.end_to_end.end_to_end_result import EndToEndResult
from utils import storage

sys.path.append(str(Path(__file__).resolve().parents[1]))
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_result(base_dir: Path) -> list[str]:
    """
    Recursively find all `.pkl` result files under the specified directory.

    Parameters
    ----------
    base_dir : Path
        The base directory to search within.

    Returns
    -------
    list[str]
        A sorted list of file paths (as strings) relative to `base_dir`
        for all `.pkl` files found.
    """
    return sorted([
        str(path.relative_to(base_dir))
        for path in base_dir.rglob("*.pkl")
    ])

def get_all_results():
    """
    Load all result objects from the project's results directory.

    This function searches the `PROJECT_ROOT / "results"` directory recursively
    for `.pkl` files, loads each result depending on its type directory, and
    returns a list of loaded result objects.

    Returns
    -------
    List[Union[CausalOrderResult, EndToEndResult]]
        A list of loaded result objects, either `CausalOrderResult` or `EndToEndResult`.

    Raises
    ------
    Exception
        If a `.pkl` file is found outside known result directories (`causal_order` or `end_to_end`).
    """
    results = find_result(PROJECT_ROOT / "results")
    ret = []
    for result in results:
        result_path = Path(result)
        # Check which directory it belongs to
        if "causal_order" in result_path.parts:
            res: CausalOrderResult = storage.load(result)
        elif "end_to_end" in result_path.parts:
            res: EndToEndResult = storage.load(result)
        else:
            raise Exception(f"Unknown result type {result_path}")
        ret.append(res)
    return ret

def print_all_results():
    """
    Print the string representations of all loaded result objects.

    This function loads all results using `get_all_results` and prints
    each result separated by two newlines.
    """
    all_results = get_all_results()
    print("\n\n".join(map(str,all_results)))

def get_results_for_dataset(dataset_name: str):
    """
    Retrieve all results associated with a specific dataset name.

    Searches the results directory for all `.pkl` files, loads them,
    and returns those whose `target_file` attribute matches `dataset_name`.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to filter results by.

    Returns
    -------
    List[Union[CausalOrderResult, CausalOrderResult]]
        A list of results matching the given dataset name. For `EndToEndResult` entries,
        only the `causal_order_result` is returned.

    Raises
    ------
    Exception
        If a `.pkl` file is found outside recognized result directories (`causal_order` or `end_to_end`).
    """
    results = find_result(PROJECT_ROOT / "results")
    ret = []
    for result in results:
        result_path = Path(result)
        # Check which directory it belongs to
        if "causal_order" in result_path.parts:
            res: CausalOrderResult = storage.load(result)
        elif "end_to_end" in result_path.parts:
            res: EndToEndResult = storage.load(result)
            res = res.causal_order_result
        else:
            raise Exception(f"Unknown result type {result_path}")
        if res.target_file != dataset_name:
            continue
        ret.append(res)
    return ret

def print_all_results_for_dataset(dataset_name: str):
    """
    Print all results associated with a specific dataset.

    Loads results filtered by the dataset name using `get_results_for_dataset`
    and prints their string representations separated by two newlines.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to filter results by.

    Returns
    -------
    None
    """
    all_results = get_results_for_dataset(dataset_name)
    print("\n\n".join(map(str,all_results)))


def main():
    print_all_results()
    with open("info.txt", "w") as f:
        f.write("\n\n".join(map(str, get_all_results())))


if __name__ == "__main__":
    main()
    #print_all_results_for_dataset("/Users/thomasjohnson/Desktop/UROP/ACORN/data/ground_truth_available/IT_monitoring/Antivirus_Activity/preprocessed_2.csv")