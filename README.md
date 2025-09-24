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

# East Germany
./paralingam_csv_emu ../../external/ACORN/data/ground_truth_available/Causal_River/East\ Germany/rivers_ts_east_germany_preprocessed.csv
# 175297 rows and 666 columns

# Flood
./paralingam_csv_emu ../../external/ACORN/data/ground_truth_available/Causal_River/Flood/rivers_ts_flood_preprocessed_dates_removed.csv
# 3005 rows and 42 columns
# Expected Order:[0, 1, 2, 3, 5, 9, 12, 14, 15, 17, 19, 22, 23, 26, 27, 28, 29, 30, 31, 34, 37, 38, 41, 10, 4, 11, 13, 16, 25, 21, 18, 33, 36, 40, 24, 20, 32, 39, 8, 35, 7, 6]
# Causal Order:  [6, 7, 8, 9, 0, 35, 5, 28, 40, 3, 19, 24, 13, 16, 1, 33, 10, 37, 36, 4, 29, 2, 15, 11, 12, 14, 17, 18, 20, 21, 22, 23, 25, 26, 27, 30, 31, 32, 34, 38, 39, 41]
# Execution Time: 1.8386 seconds
```

#### IT Monitoring (Ground Truth Available)

```bash
# Antivirus Activity
./paralingam_csv_emu ../../external/ACORN/data/ground_truth_available/IT_monitoring/Antivirus_Activity/preprocessed_3_antivirus.csv
# 1321 rows and 2 columns
# Expected Order: [12, 10, 11, 4, 9, 2, 3, 7, 8, 0, 1, 5, 6]
# Causal Order:   [5, 0, 11, 10, 7, 2, 8, 6, 1, 3, 12, 9, 4]
# Execution Time: 0.637585 seconds

# Middleware Oriented Message Activity
./paralingam_csv_emu ../../external/ACORN/data/ground_truth_available/IT_monitoring/Middleware_oriented_message_Activity/monitoring_metrics_3.csv
# 364 rows and 7 columns
# Expected Order: [0, 1, 2, 3, 5, 4, 6]
Causal Order:     [6, 5, 4, 2, 3, 0, 1]
Execution Time: 1.84444 seconds

# Storm Ingestion Activity
./paralingam_csv_emu ../../external/ACORN/data/ground_truth_available/IT_monitoring/Storm_Ingestion_Activity/storm_data_normal.csv
# 991 rows and 7 columns
# Expected Order: [0, 1, 5, 6, 7, 3, 4, 2]
Causal Order:     [5, 3, 4, 7, 6, 0, 1, 2]
Execution Time: 1.68391 seconds

# Web Activity
./paralingam_csv_emu ../../external/ACORN/data/ground_truth_available/IT_monitoring/Web_Activity/preprocessed_3.csv
# 7501 rows and 10 columns
# Expected Order: [3, 4, 8, 0, 2, 7, 5, 6, 9, 1]
# Causal Order:   [0, 3, 9, 5, 6, 7, 8, 2, 4, 1]
# Execution Time: 2.07418 seconds
```

#### S&P500 (Ground Truth Not Available)

```bash
# S&P500 Full Data
./paralingam_csv_emu_2 ../../external/ACORN/data/ground_truth_not_available/S\&P500/sp500/sp500.csv
# Causal Order: [ 416 391 157 152 190 439 193 413 289 201 25 178 293 182 435 300 180 372 198 331 224 426 312 221 141 429 219 265 214 218 336 310 371 332 109 100 422 313 243 266 90 344 95 91 236 220 164 296 388 378 437 123 334 256 82 19 39 54 112 358 15 126 375 295 415 74 258 161 92 337 179 235 346 136 195 71 42 380 122 70 110 205 443 347 400 381 306 404 268 441 87 349 252 248 240 60 0 120 1 2 3 4 5 6 7 8 9 10 11 12 13 14 16 17 18 20 21 22 23 24 26 27 28 29 30 31 32 33 34 35 36 37 38 40 41 43 44 45 46 47 48 49 50 51 52 53 55 56 57 58 59 61 62 63 64 65 66 67 68 69 72 73 75 76 77 78 79 80 81 83 84 85 86 88 89 93 94 96 97 98 99 101 102 103 104 105 106 107 108 111 113 114 115 116 117 118 119 121 124 125 127 128 129 130 131 132 133 134 135 137 138 139 140 142 143 144 145 146 147 148 149 150 151 153 154 155 156 158 159 160 162 163 165 166 167 168 169 170 171 172 173 174 175 176 177 181 183 184 185 186 187 188 189 191 192 194 196 197 199 200 202 203 204 206 207 208 209 210 211 212 213 215 216 217 222 223 225 226 227 228 229 230 231 232 233 234 237 238 239 241 242 244 245 246 247 249 250 251 253 254 255 257 259 260 261 262 263 264 267 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 290 291 292 294 297 298 299 301 302 303 304 305 307 308 309 311 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 333 335 338 339 340 341 342 343 345 348 350 351 352 353 354 355 356 357 359 360 361 362 363 364 365 366 367 368 369 370 373 374 376 377 379 382 383 384 385 386 387 389 390 392 393 394 395 396 397 398 399 401 402 403 405 406 407 408 409 410 411 412 414 417 418 419 420 421 423 424 425 427 428 430 431 432 433 434 436 438 440 442 444 445 ]
# Execution Time: 32.2842 seconds
```
---

## Bibliography

* [ParaLiNGAM Paper](https://arxiv.org/abs/2109.13993)
