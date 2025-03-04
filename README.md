# Parallelized Matrix Multiplication with NCCL

This project implements a parallelized matrix multiplication algorithm using NVIDIA's NCCL (Collective Communications Library) and CUDA. The matrix multiplication is performed across multiple GPUs, leveraging NCCL for efficient communication between the devices.

## Project Overview

The goal of this project is to accelerate the matrix multiplication operation using parallel computing across multiple GPUs. By using NCCL for collective communication, this project aims to reduce the computational time significantly, especially for large matrices, and demonstrates the use of NCCL for efficient GPU-to-GPU communication.

### Key Features:
- **CUDA for parallel matrix multiplication**: The matrix multiplication is distributed across the available GPUs using CUDA kernels.
- **NCCL for collective communication**: NCCL is used to coordinate and synchronize the parallel tasks between multiple GPUs, ensuring that the calculations are combined correctly and efficiently.
- **Fault tolerance**: The system is designed to handle large matrices and large-scale parallelism with fault tolerance and load balancing.

## Prerequisites

To build and run this project, you will need the following:
- NVIDIA GPU(s) with CUDA capability
- CUDA Toolkit installed
- NCCL library installed
- C++11 or later compiler
- MPI (Message Passing Interface) for multi-node execution (optional)
- CUDA development environment (e.g., `nvcc` compiler)

## Installation

1. **Set up the environment**:
    - Install the **CUDA Toolkit** from the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit).
    - Install **NCCL** from the [NVIDIA NCCL repository](https://developer.nvidia.com/nccl/nccl-download).
    - Install **MPI** for multi-node communication if required.

2. **Clone this repository**:
    ```bash
    git clone https://github.com/yourusername/nccl-matrix-multiplication.git
    cd nccl-matrix-multiplication
    ```

3. **Build the project**:
    Compile the project using `nvcc`:
    ```bash
    nvcc -o matmul matmul.cu -lnccl -lcudart
    ```

## Usage

### Running the Code

Once the project is built, you can run the matrix multiplication example as follows:

1. **Single GPU Example**:
   If you are running the code on a single GPU, use the following command:
   ```bash
   ./matmul
   ```
2.**Multi-GPU Example: If you have multiple GPUs and wish to parallelize the task using NCCL, use MPI:**
  ```bash
  mpirun -np 2 ./matmul
   ```
This command assumes you have two GPUs. You can adjust the number of processes (-np) according to the number of available GPUs.
