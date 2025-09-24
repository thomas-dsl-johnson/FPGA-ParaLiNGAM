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

```bash
cd UROP/FPGA-ParaLiNGAM/fpga_paralingam_c++/work_in_progress/
```

#### Causal Rivers (Ground Truth Available)

```bash
# Bavaria
./paralingam_csv_emu ../../external/ACORN/data/ground_truth_available/Causal_River/Bavaria/rivers_ts_bavaria_preprocessed.csv
# 175297 rows and 494 columns
...

# East Germany
./paralingam_csv_emu ../../external/ACORN/data/ground_truth_available/Causal_River/East\ Germany/rivers_ts_east_germany_preprocessed.csv
# 175297 rows and 666 columns
...

# Flood
# 3005 rows and 42 columns
# Expected Order: [0, 1, 2, 3, 5, 9, 12, 14, 15, 17, 19, 22, 23, 26, 27, 28, 29, 30, 31, 34, 37, 38, 41, 10, 4, 11, 13, 16, 25, 21, 18, 33, 36, 40, 24, 20, 32, 39, 8, 35, 7, 6]

# Emulation:
./paralingam_emu ../../external/ACORN/data/ground_truth_available/Causal_River/Flood/rivers_ts_flood_preprocessed.csv
# Causal Order:   [6, 7, 8, 24, 5, 3, 19, 9, 1, 31, 28, 27, 41, 18, 23, 25, 40, 30, 37, 0, 10, 29, 16, 20, 36, 12, 2, 33, 34, 32, 11, 13, 39, 21, 35, 38, 26, 22, 4, 17, 15, 14]
# Execution Time: 0.652211 seconds
# Ratio of inverted pairs:  0.017421602787456445

# Python:
 python algorithms/causal_order/new/para_lingam_causal_order_algorithm_for_fpga.py ../../ACORN/data/ground_truth_available/Causal_River/Flood/rivers_ts_flood_preprocessed.csv
# Causal Order:   [6, 7, 8, 9, 0, 35, 5, 30, 21, 38, 3, 41, 31, 22, 33, 40, 12, 1, 11, 14, 28, 24, 19, 27, 32, 37, 18, 34, 25, 16, 20, 26, 23, 2, 39, 29, 10, 36, 17, 4, 15, 13]
# Execution Time: 77.6899 seconds
# Ratio of inverted pairs:  0.018583042973286876

```

### IT Monitoring (Ground Truth Available)

#### Antivirus Activity
```
# 1321 rows and 13 columns
# Expected Order: [12, 10, 11, 4, 9, 2, 3, 7, 8, 0, 1, 5, 6]

# Emulation:
./paralingam_csv_emu ../../external/ACORN/data/ground_truth_available/IT_monitoring/Antivirus_Activity/preprocessed_2.csv
# Causal Order:   [5, 0, 11, 10, 7, 2, 8, 3, 1, 6, 12, 9, 4]
# Execution Time: 0.936734 seconds
# Ratio of inverted pairs:  0.05128205128205128

# Python:
 python algorithms/causal_order/new/para_lingam_causal_order_algorithm_for_fpga.py ../../ACORN/data/ground_truth_available/IT_monitoring/Antivirus_Activity/preprocessed_1.csv                     
# Causal Order: [11, 0, 5, 10, 7, 2, 8, 6, 1, 12, 3, 9, 4]
# Execution Time: 21.8650 seconds
# Ratio of inverted pairs:  0.057692307692307696
```
#### Middleware Oriented Message Activity
```
# 364 rows and 7 columns
# Expected Order: [0, 1, 2, 3, 5, 4, 6]

# Emulation:
./paralingam_csv_emu ../../external/ACORN/data/ground_truth_available/IT_monitoring/Middleware_oriented_message_Activity/monitoring_metrics_1.csv
# Causal Order:   [4, 3, 2, 6, 5, 0, 1]
# Execution Time: 0.60253 seconds
# Ratio of inverted pairs:  0.19047619047619047

# Python:
python algorithms/causal_order/new/para_lingam_causal_order_algorithm_for_fpga.py ../../ACORN/data/ground_truth_available/IT_monitoring/Middleware_oriented_message_Activity/monitoring_metrics_1.csv             
# Causal Order:   [4, 6, 5, 1, 0, 2, 3]
# Execution Time: 11.2405 seconds
# Ratio of inverted pairs:  0.23809523809523808
```
#### Storm Ingestion Activity
```
# 991 rows and 7 columns
# Expected Order: [0, 1, 5, 6, 7, 3, 4, 2]

# Emulation:
./paralingam_emu ../../external/ACORN/data/ground_truth_available/IT_monitoring/Storm_Ingestion_Activity/storm_data_normal.csv
Causal Order:     [5, 3, 4, 7, 6, 0, 1, 2]
Execution Time: 0.589296 seconds
Ratio of inverted pairs:  0.10714285714285714

# Python:
python algorithms/causal_order/new/para_lingam_causal_order_algorithm_for_fpga.py ../../ACORN/data/ground_truth_available/IT_monitoring/Storm_Ingestion_Activity/storm_data_normal.csv
# Causal Order: [4, 3, 5, 7, 6, 0, 1, 2]
# Execution Time: 12.4361 seconds
# Ratio of inverted pairs:  0.14285714285714285
```
#### Web Activity
```
# 7501 rows and 10 columns
# Expected Order: [3, 4, 8, 0, 2, 7, 5, 6, 9, 1]

# Emulation:
./paralingam_emu ../../external/ACORN/data/ground_truth_available/IT_monitoring/Web_Activity/preprocessed_2.csv
# Causal Order:   [0, 3, 9, 5, 6, 7, 8, 2, 4, 1]
# Execution Time: 0.620044 seconds
# Ratio of inverted pairs:  0.07777777777777778

# Python:
python algorithms/causal_order/new/para_lingam_causal_order_algorithm_for_fpga.py ../../ACORN/data/ground_truth_available/IT_monitoring/Web_Activity/preprocessed_2.csv         
# Causal Order:   [0, 3, 9, 5, 6, 7, 8, 2, 4, 1]
# Execution Time: 15.4919 seconds
# Ratio of inverted pairs:  0.07777777777777778
```

#### S&P500 (Ground Truth Not Available)

```bash
# S&P500 Full Data
./paralingam_emu ../../external/ACORN/data/ground_truth_not_available/S\&P500/sp500/sp500.csv
# Emulation:
# Cuasal Order:[ 415 390 156 151 189 438 192 412 288 200 24 177 292 181 434 330 223 220 169 421 425 361 334 370 295 186 395 362 75 10 109 342 366 264 357 90 38 307 102 407 236 313 336 152 217 106 413 332 26 387 381 444 275 396 111 53 298 94 178 161 251 335 73 160 333 77 246 8 286 419 93 365 312 279 234 309 194 173 81 227 99 219 405 108 225 265 397 18 51 125 67 317 74 320 432 374 435 190 14 377 315 318 394 114 230 41 127 341 30 47 191 91 416 130 197 360 235 388 20 311 13 126 5 240 436 226 310 343 253 233 239 159 224 110 154 344 346 60 71 409 328 213 369 299 155 144 411 167 277 321 148 287 122 135 179 297 86 294 218 44 423 42 89 72 389 263 113 85 21 410 58 255 325 188 116 427 352 32 140 31 273 142 158 210 378 274 242 164 314 115 380 339 442 260 12 107 121 316 326 168 371 306 96 257 145 19 284 283 48 303 404 203 256 340 229 25 183 7 98 356 345 261 46 211 367 175 163 134 39 258 323 322 185 269 281 198 208 391 83 34 33 429 401 216 386 62 59 259 22 222 337 2 403 414 244 296 373 439 270 363 331 16 238 36 358 245 353 0 424 70 250 35 420 308 304 293 398 193 376 354 383 149 249 180 54 28 118 43 400 65 350 215 393 143 368 441 266 69 27 82 375 132 382 348 278 271 206 63 248 153 228 267 76 364 6 17 319 182 392 282 302 49 384 209 241 428 379 443 199 372 97 137 205 268 104 204 50 45 221 418 57 11 399 124 212 301 9 252 23 359 327 272 170 437 280 300 347 139 3 146 262 123 95 101 202 150 136 305 237 131 129 349 187 40 52 92 243 78 66 355 29 431 138 176 157 88 174 285 56 231 385 324 112 37 84 117 162 105 119 120 440 351 55 207 100 195 276 417 430 61 133 4 196 338 433 402 1 64 87 214 80 147 166 426 290 172 247 68 406 15 329 165 254 201 103 141 289 232 184 171 128 79 422 291 408 ]
# Execution Time: 9.39191 seconds
python algorithms/causal_order/new/para_lingam_causal_order_algorithm_for_fpga.py ../../ACORN/data/ground_truth_not_available/S\&P500/sp500/sp500.csv
# Python:

```

#### Test

```bash
./paralingam_emu
# Ground truth: 3 -> 0, 3 -> 2, 2 -> 1, 0 -> 5, 0 -> 4, 2 -> 4
# Emulation:
# Causal Order: [3, 0, 2, 5, 4, 1]
# Execution Time: 0.594343 seconds
# Python:
# Causal Order: [3, 0, 2, 4, 5, 1]
# Execution Time: 7.5590 seconds
```

---

## Bibliography

* [ParaLiNGAM Paper](https://arxiv.org/abs/2109.13993)
