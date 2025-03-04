#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>

#define MATRIX_SIZE 1024  // Size of the matrix (1024x1024)

__global__ void matmul_kernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrix_multiply(float *A, float *B, float *C, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

int main(int argc, char* argv[]) {
    int num_gpus = 2; // Assuming 2 GPUs for parallelism
    int rank = 0;     // For simplicity, set rank to 0 for this example
    int N = MATRIX_SIZE;

    // Initialize NCCL
    ncclComm_t comm;
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, num_gpus, id, rank);

    // Allocate matrices on the host
    std::vector<float> A(N * N), B(N * N), C(N * N);

    // Initialize matrices A and B with random values
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication on each GPU in parallel
    matrix_multiply(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Synchronize and finalize NCCL
    ncclCommDestroy(comm);

    // Optionally, print some results for verification
    std::cout << "First 10 elements of the result matrix: \n";
    for (int i = 0; i < 10; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << "\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
