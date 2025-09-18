"""
When __name__ == '__main__':
    - Run the ._get_and_save_result() method of all algorithms on the selected dataset
    - Each algorithm must be in Pascal case to match its Snake case filename
"""
import importlib.util
from algorithms.generic_algorithm import GenericAlgorithm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def snake_to_pascal(snake: str) -> str:
    """
    Convert a snake_case string to PascalCase.

    Parameters
    ----------
    snake : str
        The input snake_case string, optionally ending with `.py`.

    Returns
    -------
    str
        The converted PascalCase string.
    """
    return ''.join(word.capitalize() for word in snake.replace(".py", "").split('_'))

def import_algorithm_class(module_path: Path):
    """
    Import the algorithm class from a given module path.

    Attempts to import the module specified by `module_path` and retrieve a class
    whose name matches the PascalCase version of the module name. The class must be
    a subclass of `GenericAlgorithm` to be returned.

    Parameters
    ----------
    module_path : Path
        The Python module path as a dot-separated string or `Path` (e.g., `algorithms.x.y.z`).

    Returns
    -------
    type or None
        The algorithm class if found and is a subclass of `GenericAlgorithm`, else None.
    """
    try:
        module = importlib.import_module(module_path)
        class_name = snake_to_pascal(module_path.split('.')[-1])
        cls = getattr(module, class_name, None)
        if cls and issubclass(cls, GenericAlgorithm):
            return cls
        else:
            print(f"Skipping {module_path} — class {class_name} not found or not a subclass of GenericAlgorithm.")
    except Exception as e:
        print(f"Error importing {module_path}: {e}")
    return None

def run_all_algorithms(filepath):
    """
    Run all algorithms listed in the algorithm list file on a given dataset file.

    Reads the list of algorithm module paths from `algorithms/algorithm_list.txt`,
    imports each algorithm class, creates an instance, and runs its
    `_get_and_save_result` method on the specified dataset file path.

    Parameters
    ----------
    filepath : str or Path
        The path to the dataset file to run the algorithms on.
    """
    algorithm_list_path = PROJECT_ROOT / 'algorithms' / 'algorithm_list.txt'
    with open(algorithm_list_path, 'r') as f:
        content = f.read()

    content = content.strip().strip("[]")
    module_paths = [x.strip().strip('"').strip("'") for x in content.split(',') if x.strip()]

    for module_path in module_paths:
        algo_cls = import_algorithm_class(module_path)
        if algo_cls:
            instance = algo_cls()
            print(f"\nRunning {algo_cls.__name__} from {module_path}")
            instance._get_and_save_result(filepath)


def main(filepath):
    run_all_algorithms(filepath)


if __name__ == "__main__":
    filepath = "/Users/thomasjohnson/Desktop/UROP/ACORN/data/ground_truth_not_available/S&P500/sp500_5_columns/sp500_5_columns.xlsx"
    main(filepath)
