# FPGA-ParaLiNGAM

The culmination of my UROP.
See the repositories that led us here:

External:
* [Configuration of oneAPI FPGA Runtime for DE10‐Agilex](https://github.com/ceguo/era-hello/wiki/Configuration-of-oneAPI-FPGA-Runtime-for-DE10%E2%80%90Agilex#installing-intel-fpga-add-on-for-oneapi-base-toolkit) by Ce Guo

Internal:
* [OATS](https://github.com/thomas-dsl-johnson/OATS)
* [ACORN](https://github.com/thomas-dsl-johnson/ACORN)
* [QUICHE](https://github.com/thomas-dsl-johnson/QUICHE)
* [FOOD](https://github.com/thomas-dsl-johnson/FOOD)
* [COFFEE](https://github.com/thomas-dsl-johnson/COFFEE)

---

# Project 

We have implemented the ParaLiNGAM algorithm in Python ([ACORN repository]()) and we have setup a suitable environment ([FOOD repository]()). The aim is to implement the algorithm with a successful report. 

## FPGA ParaLiNGAM Main

###### Python Implementation:
* This implementation is a Python equivalent of the C++ code.
* Algorithmic Faithfulness:
	* Parallel Root Finding: Uses Python's multiprocessing.Pool to parallelize the score calculations, logically mirroring the DPC++ "Scatter-Reduce" approach.
	* Messaging/Comparison Reduction: The worker_task correctly updates scores for both variables involved in a comparison simultaneously.
	* Efficient Covariance Update: The _update_covariance_matrix function uses vectorized NumPy operations to implement the same efficient update formula as the C++ version, avoiding costly recalculations.
 	* No Threshold Mechanism: Like the C++ version it mirrors, this implementation does not include the thresholding logic.

Run:
```bash
    python para_lingam_causal_order_algorithm_for_fpga.py
```

###### C++ Implementation:
* This implementation is a version of the core ParaLiNGAM logic, specifically tailored for an FPGA target.
* Algorithmic Faithfulness:
	* Parallel Root Finding: Correctly uses a "Scatter-Reduce" pattern in para_find_root, which is an excellent, FPGA-friendly way to parallelize the score calculations without inefficient atomic operations.
	* Messaging/Comparison Reduction: The scatter-reduce kernel correctly implements the messaging mechanism by calculating scores for both variables (i and j) from a single comparison, effectively halving the workload.
	* Efficient Covariance Update: It faithfully implements the paper's mathematical simplification for updating the covariance matrix between iterations in the update_covariance kernel, which is a critical performance optimization.
	* No Threshold Mechanism: This feature is deliberately omitted. The complex, stateful logic of thresholding is not well-suited for a simple, high-throughput FPGA kernel architecture. The current design prioritizes hardware efficiency over implementing this specific software-level optimization.

FPGA Emulation:
```
icpx -fintelfpga -DFPGA_EMULATOR main.cpp ParaLingam.cpp -o paralingam_faithful_emu
```

FPGA Hardware Report:
```bash
icpx -fintelfpga -Xssource-report main.cpp ParaLingam.cpp -o paralingam_faithful_report
```

## Part 0

We will do a minimal emulation test.

```bash
# Compile for the FPGA Emulator
icpx -fintelfpga -DFPGA_EMULATOR test_minimal.cpp -o test_minimal.fpga_emu

# Run
./test_minimal.fpga_emu

# Output
Running on device: Intel(R) FPGA Emulation Device
Minimal test successful. Result: 30
```

## Part 1

We will do a minimal hardware report test.

```bash
# Compile for FPGA Optimisation Report
icpx -fintelfpga -DFPGA_HARDWARE step1_minimal_hw.cpp -Xshardware -fsycl-link=early -Xstarget=Agilex7 -o minimal_report.a

# Output:
# Successfully Generates reports

# Copu to local:
exit
cd ~/Downloads
scp -r ccgpu4:/homes/tdj23/UROP/FPGA-ParaLiNGAM/part_1/ .
```


## Part 2

We will compile the simplest part of the algorithm—the standardise_data kernel—to validate the toolchain and generate a baseline report.

The main function should initialise a SYCL queue, create a sample Matrix, wrap it in a SYCL buffer, and call only the standardize_data kernel.

```bash
# Emulate
icpx -fintelfpga -DFPGA_EMULATOR part_2.cpp -o part_2.fpga_emu
./part_2.fpga_emu

# Output:
# Running on device: Intel(R) FPGA Emulation Device
# Running standardize_data kernel...
# Kernel execution finished.

# Compile for FPGA optimisation report
icpx -fintelfpga -DFPGA_HARDWARE part_1.cpp -Xshardware -fsycl-link=early -o part_1_report.a

# Report overview
Single system kernel:
System Viewer: Shows one simple kernel system connected to global memory. This is baseline hardware. We just do standardise data.
Area Estimates: Establishes the minimum resource cost (ALUTs, FFs) for the simplest part of our algorithm. 

# Copy to local
exit
cd ~/Downloads
scp -r 34.89.44.85:/home/thomasjohnson/COFFEE/container_assets/Project/para_lingam/part_2 .
```

## Part 3

```bash
# Report
icpx -fintelfpga -DFPGA_HARDWARE part_3.cpp -Xshardware -fsycl-link=early -Xstarget=Agilex7 -o part3_report.a

# Report Overview
Integrating a Second Kernel:
System Viewer: Now shows two separate kernel systems. This visually confirms the host-driven architecture where data flows between kernels via off-chip global memory. 
Area Estimates: The total resource usage increases, showing the cost of the added logic for the second kernel. 
```

## Part 4

```bash
# Report
icpx -fintelfpga -DFPGA_HARDWARE part_4.cpp -Xshardware -fsycl-link=early -Xstarget=Agilex7 -o part_4_report.a

# Overview
Structuring the Core Logic (Simplified para_find_root)
System Viewer: Shows three separate kernel systems. The third system for para_find_root is visibly more complex
Area Estimates: The area cost for para_find_root show its structural complexity, but without the high DSP usage from complex maths. 
Loop Analysis: Reveals the baseline II for the para_find_root loops. An II > 1 indicates a structural bottleneck. We have II NA - to do look into this. 

```

## Part 5

```bash
# Report:
icpx -fintelfpga -DFPGA_HARDWARE part_5.cpp -Xshardware -fsycl-link=early -Xstarget=Agilex7 -o part_5.a

Implementing the Full Core Logic
Area Estimates: Shows a significant increase in DSP block usage for the para_find_root kernel, reflecting the cost of the complex entropy calculations. 
Loop Analysis: The II of the main loop in para_find_root increases significantly. The report's Details pane will point to a long latency calculation as the bottleneck. 
```


## Part 6

```bash
# Source environment if in a new terminal
source /opt/intel/oneapi/setvars.sh --force
icpx -fintelfpga -DFPGA_HARDWARE part_6.cpp -Xshardware -fsycl-link=early -Xstarget=Agilex7 -o part_6.a

# Full Host-Driven Algorithm
```

##

To do:
1. Refactor to a Single-Kernel Design
   Eliminate the main performance bottleneck by moving the entire algorithm loop from the host CPU onto the FPGA itself.
   Rewrite the code to use a single, large kernel that contains the main for loop.
   All intermediate data stored in the FPGA's fast on-chip memory instead of being passed back to the host after each iteration.
   Host will only be responsible for starting the kernel once and collecting the final result.
2. True dataflow pipeline, which is the `ideal architecture for FPGAs`.
   Maximize parallelism and improve the clock frequency (fMAX) of the final design.
   We will break the single large kernel back into our smaller, specialized kernels. However, instead of calling them sequentially, we will launch them all to run concurrently and connect them using sycl::ext::intel::pipe objects. Data will stream directly from one kernel to the next without ever touching on-chip RAM blocks.

---

## Enable oneAPI

```bash
source /mnt/ccnas2/bdp/opt/Intel/oneapi/setvars.sh
		
alias quartus='/mnt/ccnas2/bdp/opt/Intel/intelFPGA_pro/21.4/quartus/bin/quartus' 
		
export QSYS_ROOTDIR='/mnt/ccnas2/bdp/opt/Intel/intelFPGA_pro/21.4/quartus/sopc_builder/bin' 
		
alias quartus_shell='/mnt/ccnas2/bdp/opt/Intel/intelFPGA_pro/21.4/embedded/embedded_command_shell.sh' 
		
export PATH=/mnt/ccnas2/bdp/Intel/intelFPGA_pro/21.4/hld/bin:/mnt/ccnas2/bdp/opt/Intel/intelFPGA_pro/21.4/quartus/bin/:$PATH
```

```bash
export XILINX_XRT=/opt/xilinx/xrt
export PATH=$XILINX_XRT/bin:$PATH
export LD_LIBRARY_PATH=$XILINX_XRT/lib:$LD_LIBRARY_PATH
```

---

## Outputs



---

## Bibliography

* [ParaLiNGAM Paper](https://arxiv.org/abs/2109.13993)
