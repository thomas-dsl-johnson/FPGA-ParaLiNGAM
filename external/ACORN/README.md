# ACORN - Assess Causal Order Results Neatly
The ACORN repository provides a neat and simple way to assess and visualise the differences between different algorithms that generate causal orders from data

# Table of Contents

1. [About ACORN](#acorn---assess-causal-order-results-neatly)  
2. [Usage](#-usage)
3. [Getting Started](#-getting-started)  
 2.1 [Python Version](#python-version)  
 2.2 [Installation](#1-installation)  
  2.2.1 [Clone the Repository](#1a-clone-the-repository)  
  2.2.2 [Complete Setup of Causal River Datasets](#1b-complete-setup-of-causal-river-datasets)  
  2.2.3 [Setup of Your Own Datasets](#1c-setup-of-your-own-datasets)   
4. [Repository Structure](#-repository-structure)  
 4.1 [algorithms/](#algorithms)  
 4.2 [data/](#data)  
 4.3 [external/](#external)  
 4.4 [results/](#results)  
 4.5 [utils/](#utils)  
5. [Notes](#-notes)

----
## 📊 Usage

To test your have a working installation of the repository. Try running the following. It may take quite some time to run.
```bash
python run.py
```
Or do the following step-by-step process.

```bash
python generate_list_of_algorithms.py
# This file finds the algorithms within the algorithms/ directory and makes a list.
# The output is written to the algorithms/ directory in algorithm_list.txt.
python run_all_algorithms_on_dataset.py
# Run all algorithms in the algorithm_list.txt on the selected dataset
python print_results.py
# Run the main method of print_results.py
```
For more advanced use cases or to make your own implementations you can find more detail in the relevant [Utilities section](#utils). 

For setting up your own datasets see them relevant [Installation Section](#1c-setup-of-your-own-datasets).

-----
## 🚀 Getting Started

### Python Version

The repository was tested using the following version of Python.

```
Python 3.12.4
```
You can check your own Python version by running this command in your terminal:
```bash
python --version
```

### 1\. Installation

#### 1\.a Clone the repository
To get started, clone the repository and install the necessary Python packages using the `requirements.txt` file. (It is recommended to install packages and run all scripts within a virtual environment to avoid dependency conflicts.)
```bash
git clone https://github.com/thomas-dsl-johnson/ACORN.git
cd ACORN
pip install -r requirements.txt
```
We now have the following file structure. To see a full explanation of the file structure, go to [📂 Repository Structure](https://github.com/thomas-dsl-johnson/ACORN?tab=readme-ov-file#-repository-structure).
```
.
├── algorithms/ ...
├── data/ ...
├── external/ ...
├── README.md
├── requirements.txt
├── run.py
└── utils/ ...
```
#### 1\.b Complete setup of Causal River datasets
If you do not require the Causal_River Bavaria and East Germany datasets, you can skip this step. Let's look at the `data/` directory:
```
data
├── ground_truth_available
│   ├── Causal_River
│   │   ├── Bavaria/ ...
│   │   ├── East Germany/ ...
│   │   └── Flood/ ...
│   └── IT_monitoring
│       ├── Antivirus_Activity/ ...
│       ├── Middleware_oriented_message_Activity/ ...
│       ├── Storm_Ingestion_Activity/ ...
│       └── Web_Activity/ ...
└── ground_truth_not_available
    └── S&P500
        ├── sp500/ ...
        └── sp500_5_columns/ ...
```
The `Causal_River/` directory has the `Flood/` dataset directory but is missing the `Bavaria/` and `East Germany/` directories that appear in its [repository structure explanation](https://github.com/thomas-dsl-johnson/ACORN?tab=readme-ov-file#data). This is because the `CausalRiverBavaria` and `CausalRiverEastGermany` datasets are too large for this repository. We will need to download them from the original [CausalRivers GitHub repository](https://github.com/CausalRivers/causalrivers).  
```bash
# 1. Clone the following submodules: 1. Causal River, 2. Causal Graph Recovery from Casual Order 
git submodule update --init --recursive
cd external/causal_rivers
# 2. Follow the install steps for Causal River
./install.sh
conda activate causalrivers
python 0_generate_datsets.py
# 3. Create and Populate the Bavaria and East Germany Directories inside data/ground_truth_available/Causal_river/
cd ../..
mkdir data/ground_truth_available/Causal_River/Bavaria
mkdir data/ground_truth_available/Causal_River/East\ Germany              ─╯
cp external/causal_rivers/product/rivers_bavaria.p data/ground_truth_available/Causal_River/Bavaria
cp external/causal_rivers/product/rivers_east_germany.p data/ground_truth_available/Causal_River/East\ Germany
cp external/causal_rivers/product/rivers_ts_bavaria.csv data/ground_truth_available/Causal_River/Bavaria
cp external/causal_rivers/product/rivers_ts_east_germany.csv data/ground_truth_available/Causal_River/East\ Germany
```
We now have:
```
Causal_River
├── Bavaria
│   ├── rivers_bavaria.p
│   └── rivers_ts_bavaria.csv
├── East Germany
│   ├── rivers_east_germany.p
│   └── rivers_ts_east_germany.csv
└── Flood/ ...
```
We must now preprocess the .csv files using the [Causal Graph Recovery from Causal Order Repository](https://github.com/jultrishyyy/Recover-Causal-Graph-from-Causal-Order/tree/50e7f0a7b06cca6623de99a4b467a71f70deca1b?tab=readme-ov-file#-repository-structure). 
```bash
cd external/recover_causal_graph_from_causal_order/generate_ground_truth
```
Using your favourite editor, change the following constants in `process_causalriver.py`
```python
ROOT_DIR = os.getcwd()
DATA_PATH = os.path.join(ROOT_DIR, "data/ground_truth_available/Causal_River", "East Germany")
input_filename = DATA_PATH + "/rivers_ts_east_germany.csv"
output_filename = DATA_PATH + "/rivers_ts_east_germany_preprocessed.csv"
```
Run the file.
```bash
python process_causalriver.py
```
Now repeat for Bavaria.
```python
ROOT_DIR = os.getcwd()
DATA_PATH = os.path.join(ROOT_DIR, "data/ground_truth_available/Causal_River", "Bavaria")
input_filename = DATA_PATH + "/rivers_ts_bavaria.csv"
output_filename = DATA_PATH + "/rivers_ts_bavaria_preprocessed.csv"
```
Run the file.
```bash
python process_causalriver.py
```
We now have:
```
Causal_River
├── Bavaria
│   ├── rivers_bavaria.p
│   ├── rivers_ts_bavaria.csv
│   └── rivers_ts_bavaria_preprocessed.csv
├── East Germany
│   ├── rivers_east_germany.p
│   ├── rivers_ts_east_germany.csv
│   └── rivers_ts_east_germany_preprocessed.csv
└── Flood/ ...
```
Now we will create the `summary_matrix.npy` files for Bavaria and East Germany. We will edit `generate_causalriver_summary_matrix.py` 
```python
ROOT_DIR = os.getcwd()
DATA_PATH = os.path.join(ROOT_DIR, "data/ground_truth_available/Causal_River", "East Germany")
input_data_filename = DATA_PATH + "/rivers_ts_east_germany_preprocessed.csv" # Make sure this path is correct
input_label_filename = DATA_PATH + "/rivers_east_germany.p" # Ground truth graph data
output_matrix_filename = DATA_PATH + '/summary_matrix.npy'
```
Then run the file.
```bash
python generate_causalriver_summary_matrix.py
```
Now, repeat for Bavaria.
```python
ROOT_DIR = os.getcwd()
DATA_PATH = os.path.join(ROOT_DIR, "data/ground_truth_available/Causal_River", "Bavaria")
input_data_filename = DATA_PATH + "/rivers_ts_bavaria_preprocessed.csv" # Make sure this path is correct
input_label_filename = DATA_PATH + "/rivers_bavaria.p" # Ground truth graph data
output_matrix_filename = DATA_PATH + '/summary_matrix.npy'
```
Then run the file.
```bash
python generate_causalriver_summary_matrix.py
```
We now have:
```
Causal_River
├── Bavaria
│   ├── rivers_bavaria.p
│   ├── rivers_ts_bavaria.csv
│   ├── rivers_ts_bavaria_preprocessed.csv
│   └── summary_matrix.npy
├── East Germany
│   ├── rivers_east_germany.p
│   ├── rivers_ts_east_germany.csv
│   ├── rivers_ts_east_germany_preprocessed.csv
│   └── summary_matrix.npy
└── Flood/ ...
```

For completeness, we now we need to generate the `causal_order.txt` files. We will edit `generate_order_from_matrix.py`
```python
ROOT_DIR = os.getcwd()
###
### Unchanged Code
###
DATA_PATH = os.path.join(ROOT_DIR, "data/ground_truth_available/Causal_River", "East Germany")
```
Then run the file.
```bash
python generate_order_from_matrix.py
```
Then change for Bavaria.
```python
ROOT_DIR = os.getcwd()
###
### Unchanged Code
###
DATA_PATH = os.path.join(ROOT_DIR, "data/ground_truth_available/Causal_River", "Bavaria")
```
Then run the file.
```bash
python generate_order_from_matrix.py
```

We are done. The `Causal_River` file structure should now match as it appears in the [Repository Structure](https://github.com/thomas-dsl-johnson/ACORN?tab=readme-ov-file#data)

#### 1\.c Setup of your own datasets

If you have your own dataset then place it in `data/` and read the instructions below.
 
#### 1\.c.i. Your data has a ground truth:
  
Ensure the following data is correctly formatted and placed appropriately within the `data/ground_truth_available` directory:
  * **Causal Order**: The causal order of variables should be stored in a `causal_order.txt` file as a Python list.
  * **Ground Truth Summary Matrix**: The ground truth summary matrix must be in a `summary_matrix.npy` file, saved as a NumPy array.
  * **Dataset**: Your dataset should be in a `.csv` file.

See below for how the file structure, `causal_order.txt`, and `summary_matrix.npy` should appear. 

#### 1\.c.ii. Your data does not have a ground truth

Ensure the following data is correctly formatted and placed appropriately within the `data/ground_truth_not_available` directory:
  * **Dataset**: Your dataset should be in a `.csv` file.

Then run `utils/generate_data_when_ground_truth_not_available.py` to generate the `causal_order.txt`, and `summary_matrix.npy` files. Here, we are using DirLiNGAM to create a 'synthetic' ground truth. Replace the argument with the location of your datset.

```bash
cd utils
python generate_data_when_ground_truth_not_available.py data/ground_truth_not_available/Dataset/data.csv
```

See below for check how the file structure, `causal_order.txt`, and `summary_matrix.npy` should appear.

**File Structure Example:**
```
data/
├── ground_truth_available
│   └── Dataset1/
│       ├── causal_order.txt
│       ├── dataset1.csv
│       └── summary_matrix.npy
│       ...
└── ground_truth_not_available
    └── Dataset2/
        ├── causal_order.txt
        ├── dataset2.csv
        ├── model.pkl
        └── summary_matrix.npy
        ...
```

**`causal_order.txt` Example:**
```
[0, 1, 2, 3, 5, 4, 6]
```

**`summary_matrix.npy` Example:**
```python
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 0, 1, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 0, 1, 0]]
```

----
## 📂 Repository Structure

```
ACORN   
├── README.md
├── algorithms/ ...
├── data/ ...
├── external/ ...
├── requirements.txt
├── results/ ...
├── run.py
└── utils/ ...
```

#### `algorithms/`

```
algorithms
├── algorithm_list.txt
├── causal_order
│   ├── causal_order_result.py
│   ├── generic_causal_order_algorithm.py
│   ├── new
│   │ ├── direct_lingam_causal_order_algorithm_adding_nodes_in_batches_of_two.py
│   │ ├── direct_lingam_causal_order_algorithm_adding_nodes_in_batches_of_x.py
│   │ ├── direct_lingam_causal_order_algorithm_lookup.py
│   │ ├── direct_lingam_causal_order_algorithm_no_updates.py
│   │ ├── direct_lingam_causal_order_algorithm_threshold.py
│   │ └── para_lingam_causal_order_algorithm.py
│   └── original
│       └── direct_lingam_causal_order_algorithm.py
├── end_to_end
│   ├── end_to_end_result.py
│   ├── generic_end_to_end_algorithm.py
│   ├── new
│   │ ├── generic_algorithm_with_causal_graph_recovery_from_causal_order_.py
│   │ └── with/ ...
│   └── original
│       └── direct_lingam_end_to_end_algorithm.py
└── generic_algorithm.py
```
`generic_algorithm.py:` An abstract base class that defines the common interface for all algorithms, including methods for running the algorithm and handling results.

`causal_order/`: Contains algorithms that only determine the causal order of variables.
    
* `generic_causal_order_algorithm.py`: A generic class for causal order algorithms.
* `original/direct_lingam_causal_order_algorithm.py`: The standard implementation of the DirectLiNGAM causal ordering phase.
* `new/`: Contains variations of the causal order algorithm, such as adding nodes in batches.

`end_to_end/`: Contains algorithms that perform full causal graph discovery.
* `generic_end_to_end_algorithm.py`: A generic class for end-to-end algorithms. 
* `original/direct_lingam_end_to_end_algorithm.py`: An implementation that uses the lingam library to perform the full DirectLiNGAM analysis.
* `new/generic_algorithm_with_causal_graph_recovery_from_causal_order_.py`: Runs a two-step approach that first finds the causal order using a subclass of generic_causal_order_algorithm and then recovers the causal graph using methods from the [Recover-Causal-Graph-from-Causal-Order](https://github.com/jultrishyyy/Recover-Causal-Graph-from-Causal-Order/tree/main) dataset.

#### `data/`
```
data
├── ground_truth_available
│   ├── Causal_River
│   │   ├── Bavaria
│   │   │   ├── causal_order.txt
│   │   │   ├── rivers_bavaria.p
│   │   │   ├── rivers_ts_bavaria.csv
│   │   │   ├── rivers_ts_bavaria_preprocessed.csv
│   │   │   └── summary_matrix.npy
│   │   ├── East Germany
│   │   │   ├── causal_order.txt
│   │   │   ├── rivers_east_germany.p
│   │   │   ├── rivers_ts_east_germany.csv
│   │   │   ├── rivers_ts_east_germany_preprocessed.csv
│   │   │   └── summary_matrix.npy
│   │   └── Flood
│   │       ├── causal_order.txt
│   │       ├── rivers_flood.p
│   │       ├── rivers_ts_flood.csv
│   │       ├── rivers_ts_flood_preprocessed.csv
│   │       ├── rivers_ts_flood_preprocessed_dates_removed.csv
│   │       └── summary_matrix.npy
│   └── IT_monitoring
│       ├── Antivirus_Activity
│       │   ├── causal_graph_label.png
│       │   ├── causal_order.txt
│       │   ├── preprocessed_1.csv
│       │   ├── preprocessed_2.csv
│       │   ├── structure.txt
│       │   └── summary_matrix.npy
│       ├── Middleware_oriented_message_Activity
│       │   ├── causal_graph_label.png
│       │   ├── causal_order.txt
│       │   ├── monitoring_metrics_1.csv
│       │   ├── monitoring_metrics_2.csv
│       │   ├── structure.txt
│       │   └── summary_matrix.npy
│       ├── Storm_Ingestion_Activity
│       │   ├── causal_graph_label.png
│       │   ├── causal_order.txt
│       │   ├── storm_data_normal.csv
│       │   ├── structure.txt
│       │   └── summary_matrix.npy
│       └── Web_Activity
│           ├── causal_graph_label.png
│           ├── causal_order.txt
│           ├── preprocessed_1.csv
│           ├── preprocessed_2.csv
│           ├── structure.txt
│           └── summary_matrix.npy
└── ground_truth_not_available
    └── S&P500
        ├── sp500
        │   ├── causal_order.txt
        │   ├── model.pkl
        │   ├── sp500.csv
        │   └── summary_matrix.npy
        └── sp500_5_columns
            ├── causal-graph
            ├── causal-graph.pdf
            ├── causal_order.txt
            ├── model.pkl
            ├── sp500_5_columns.xlsx
            └── summary_matrix.npy
```

This directory contains the datasets. Each dataset has its own subfolder, which includes the raw data and the corresponding ground truth files. The repository includes:

&nbsp;&nbsp;&nbsp;&nbsp;`ground_truth_available/`
* IT Monitoring Data:  Source: [Case\_Studies\_of\_Causal\_Discovery](https://github.com/ckassaad/Case_Studies_of_Causal_Discovery_from_IT_Monitoring_Time_Series)
* CausalRiver Datasets: Source: [CausalRivers](https://github.com/CausalRivers/causalrivers). For the Bavaria and East Germany data you must complete [step 1b](https://github.com/thomas-dsl-johnson/ACORN?tab=readme-ov-file#1b-complete-setup).

&nbsp;&nbsp;&nbsp;&nbsp;`ground_truth_not_available/`
* S&P500 Data


#### `external/`

```
external
├── causal_rivers/ ...
└── recover_causal_graph_from_causal_order/ ...
```
We have 2 submodules: [Causal Rivers](https://github.com/CausalRivers/causalrivers) and [Causal Graph Recovery from Causal Order Repository](https://github.com/ckassaad/Case_Studies_of_Causal_Discovery_from_IT_Monitoring_Time_Series). To clone the submodules, run the following code snippet. This is done during installation step 1b.
```bash
git submodule update --init --recursive
```

#### `results/`
```
results
├── causal_order/ ...
└── end_to_end/ ...
```

This folder stores the outputs of analysis. 

#### `utils/`

```
utils
├── compare_results.py
├── generate_data_when_ground_truth_not_available.py
├── generate_list_of_algorithms.py
├── print_results.py
├── run_all_algorithms_on_dataset.py
└── storage.py
```

This directory contains utility scripts.

&nbsp;&nbsp;&nbsp;&nbsp;`compare_results.py` A set of functions that compare two results.

&nbsp;&nbsp;&nbsp;&nbsp;`generate_data_when_ground_truth_not_available.py` This file loads data from a .csv or .xlsx file
It then runs DirectLiNGAM and save the results of this to the same directory as the input file.

&nbsp;&nbsp;&nbsp;&nbsp;`generate_list_of_algorithms.py` This file finds the algorithms within the algorithms/ directory and makes a list.
The output is written to the algorithms/ directory in algorithm_list.txt.

&nbsp;&nbsp;&nbsp;&nbsp;`print_results.py` Find the results within the algorithms/ directory and prints a summary of each one.

&nbsp;&nbsp;&nbsp;&nbsp;`run_all_algorithms_on_dataset.py` Run the `._get_and_save_result()` method of all algorithms on the selected dataset

&nbsp;&nbsp;&nbsp;&nbsp;`storage.py` This module handles the serialisation of data and deserialisation of results.


-----
## 📝 Notes

  * Ensure that the number of variables in your dataset matches the dimensions of the summary matrix.
  * For large datasets (more than 15 variables), such as `CausalRiverFlood`, visualising the full causal graph is not recommended as it can become cluttered and difficult to interpret.
  * Thank you to Zhao Tong ([@jultrishyyy](https://github.com/jultrishyyy)) for the [Causal Graph Recovery from Causal Order Repository](https://github.com/jultrishyyy/Recover-Causal-Graph-from-Causal-Order/tree/50e7f0a7b06cca6623de99a4b467a71f70deca1b?tab=readme-ov-file#causal-graph-recovery-from-causal-order) and its detailed README 
  * For any issues or questions, please open an issue on the repository's issue tracker.
