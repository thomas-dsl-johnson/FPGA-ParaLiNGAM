"""
This module handles the serialisation of data and deserialisation of results.
By default, this module saves to the results/ directory
"""
import pickle
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

def save(obj: Any, filename: str, save_to_default=True):
    """
    Save an object to a pickle file in the output directory.

    Parameters
    ----------
    obj : Any
        The Python object to be serialized and saved.
    filename : str
        The name of the file
    """

    if save_to_default:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        full_file_path = os.path.join(RESULTS_DIR, filename)
    else:
        full_file_path = filename
    os.makedirs(os.path.dirname(full_file_path), exist_ok=True)
    with open(full_file_path, 'wb') as file:
        pickle.dump(obj, file)
    print(f'Saved {full_file_path}')


def load(filename: str) -> Any:
    """
    Load a Python object from a pickle file in the output directory.

    Parameters
    ----------
    filename : str
        The name of the file to load (should match a file saved in the output directory).

    Returns
    -------
    obj : Any
        The Python object loaded from the pickle file.
    """
    full_file_path = os.path.join(RESULTS_DIR, filename)
    with open(full_file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj


def exists(filename: str):
    """
    Check if a file exists in the output directory.

    Parameters
    ----------
    filename : str
        The name of the file to check.

    Returns
    -------
    exists : bool
        True if the file exists, False otherwise.
    """
    full_file_path = os.path.join(RESULTS_DIR, filename)
    return os.path.exists(full_file_path)