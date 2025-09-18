"""
This function loads data from a .csv or .xlsx file
It then runs DirectLiNGAM and save the results of this to the same directory as the input file
input: .csv or .xlsx
output: causal_order.txt, summary_matrix.npy, model.pkl and (optionally) causal_graph.pdf

Important: The file must be run with an argument for the target .csv file
To run this file:
cd utils
python generate_data_when_ground_truth_not_available.py data/ground_truth_not_available/Dataset/data.csv
"""
import sys
import os
import lingam
import pandas as pd
import numpy as np
import storage
from lingam.utils import make_dot

if "__main__" == __name__:
    if len(sys.argv) != 2:
        raise Exception("Target .csv file required as argument e.g. data/ground_truth_not_available/Dataset/data.csv")
    filepath = os.path.dirname(os.getcwd()) + '/' + sys.argv[1]
    file_extension = os.path.splitext(filepath)[1].lower()
    if file_extension == '.csv':
        X = pd.read_csv(filepath)
    elif file_extension in ['.xls', '.xlsx']:
        X = pd.read_excel(filepath)
    else:
        raise ValueError(
            f"Unsupported file type: '{file_extension}'. "
            "Only .csv, .xls, and .xlsx files are currently supported."
        )
    print("Creating model")
    model = lingam.DirectLiNGAM()
    model.fit(X)
    print("Model created. Saving data")
    dest = "../" + os.path.dirname(sys.argv[1])
    # Write causal_order.txt
    with open(dest + "/causal_order.txt", "w") as file:
        file.write('[')
        file.write(", ".join(map(str,model.causal_order_)))
        file.write(']')
    print("Causal Order Saved to " + dest + "/causal_order")
    # Write summary_matrix.npy
    np.save(dest + "/summary_matrix.npy", model.adjacency_matrix_)
    print("Summary matrix saved to " + dest + "/summary_matrix")
    # Write pickle file
    storage.save(model, dest + "/model.pkl", save_to_default=False)
    print("Model saved to " + dest + "/model.pkl")
    # Write .png if size is small enough
    if model.adjacency_matrix_.shape[0] <= 15:
        dot = make_dot(model.adjacency_matrix_)
        dot.render(dest + '/causal-graph')
        print(".pdf saved to " + dest + '/causal-graph')
