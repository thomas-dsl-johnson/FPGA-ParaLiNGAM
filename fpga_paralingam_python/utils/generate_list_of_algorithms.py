"""
This file finds the algorithms within the algorithms/ directory and makes a list.
The output is written to the algorithms/ directory in algorithm_list.txt
"""
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
"""
This file finds the algorithms within the algorithms/ directory and makes a list.
The output is written to the algorithms/ directory in algorithm_list.txt
"""
import os
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = PROJECT_ROOT / "algorithms"

def is_algorithm(file_path: Path) -> bool:
    """
    Check if a file is a Python algorithm implementation.

    Parameters
    ----------
    file_path : Path
        Path to the file to check.

    Returns
    -------
    bool
        True if the file is a Python algorithm implementation, False otherwise.
    """
    name = file_path.name
    return (
        file_path.suffix == ".py"
        and "generic" not in name.lower()
        and "__init__" not in name.lower()
        and "__pycache__" not in file_path.parts
        and name.lower() not in {"end_to_end_result.py", "causal_order_result.py"}
    )

def path_to_module(path: Path) -> str:
    """
    Convert a file path to a Python module path.

    Converts a path such as `algorithms/x/y/z.py` into the equivalent
    dotted module path `algorithms.x.y.z`.

    Parameters
    ----------
    path : Path
        Path to the Python source file.

    Returns
    -------
    str
        The corresponding Python module path.
    """
    return ".".join(path.with_suffix("").parts)

def find_algorithm_modules(base_dir: Path) -> List[str]:
    """
    Recursively find all algorithm Python files and return them as importable module paths.

    This function searches within the given directory for `.py` files that qualify
    as algorithm implementations (based on `is_algorithm`), and converts their paths
    to Python module import paths.

    Parameters
    ----------
    base_dir : Path
        The base directory to search within.

    Returns
    -------
    List[str]
        A sorted list of module paths corresponding to algorithm files.
    """
    return sorted([
        path_to_module(path.relative_to(PROJECT_ROOT))
        for path in base_dir.rglob("*.py")
        if is_algorithm(path)
    ])


def main():
    module_paths = find_algorithm_modules(ALGORITHMS_DIR)
    for module in module_paths:
        print(module)
    output_file = ALGORITHMS_DIR / "algorithm_list.txt"
    with open(output_file, "w") as f:
        f.write("[\n")
        for module in module_paths:
            f.write(f'    "{module}",\n')
        f.write("]\n")


if __name__ == "__main__":
    main()

