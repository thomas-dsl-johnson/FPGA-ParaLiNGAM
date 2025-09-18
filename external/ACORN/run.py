"""
This is the entry point for running ACORN - Assess Causal Order Results Neatly
See the README.md for more information
"""
import sys
from pathlib import Path
from utils import generate_list_of_algorithms, run_all_algorithms_on_dataset, print_results

PROJECT_ROOT = Path(__file__).resolve().parents[0]

if __name__ == "__main__":
    file_path = None
    print(sys.argv)
    if len(sys.argv) > 1:
        file_path = PROJECT_ROOT / "data" / sys.argv[0]
    else:
        file_path = PROJECT_ROOT / "data/ground_truth_not_available/S&P500/sp500_5_columns/sp500_5_columns.xlsx"
    generate_list_of_algorithms.main()
    print("Generated list of algorithms")
    run_all_algorithms_on_dataset.main(file_path)
    print("Ran all algorithms on dataset")
    print_results.main()
    print("Finished printing results")


