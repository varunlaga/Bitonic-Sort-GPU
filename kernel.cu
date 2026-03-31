#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
// ==== DO NOT MODIFY CODE ABOVE THIS LINE ====

#define DTYPE short
// Add any additional #include headers or helper macros needed
#define SENTINEL 32767

// Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).

// This kernel performs the bitonic sort algorithm and works directly on data in global memory
__global__ void bitonic_Sort_Kernel(short *data, int j, int k, int n) {
    // Calculate the global index of the current thread
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Calculate the index of the element to compare with the partner index
    // The XOR operation efficiently determines the partner element for the comparison
    unsigned int ixj = idx ^ j;

    // Check bounds to ensure both threads in a pair are within the valid data range
    // The ixj > idx check ensures that each pair is processed by only one thread
    if (ixj > idx && idx < n && ixj < n) {
        // The (idx & k) == 0 condition determines the sorting direction (ascending or descending) based on current stage of bitonic sort
        if ((idx & k) == 0) {
            // Sort in ascending order
            if (data[idx] > data[ixj]) {
                // Swap the elements if they are in the wrong order
                short temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            // Sort in descending order
            if (data[idx] < data[ixj]) {
                // Swap the elements if they are in the wrong order
                short temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

// This optimized kernel uses shared memory to perform the bitonic sort algorithm
__global__ void bitonic_Sort_Kernel_shared(short *data, int j, int k, int n) {
    // Declare a dynamically-sized shared memory array
    extern __shared__ short shared_data[];

    // Get the thread's local ID within the block
    unsigned int tid = threadIdx.x;

    // Get the thread's global index
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Load data from global memory to shared memory and this is a coalesced read operation
    if (idx < n) {
        shared_data[tid] = data[idx];
    } else {
        // Use a sentinel value for out-of-bounds threads to prevent garbage values from being sorted
        shared_data[tid] = SENTINEL;
    }
    // Synchronize all threads in the block to ensure all data is loaded
    __syncthreads();

    // Perform multiple bitonic sort steps entirely within shared memory
    // This loop handles all sort stages that fit within the block size
    for (int current_j = j; current_j > 0; current_j >>= 1) {
        // Exit the loop if the sorting step j is larger than the block size and indicates the step must be handled at the block level
        if (current_j >= blockDim.x) { 
            break;
        }

        // Calculate the partner index within the shared memory array
        unsigned int ixj = tid ^ current_j;

        // Check if the partner index is within the block boundaries
        if (ixj > tid && ixj < blockDim.x) {
            // The sorting direction is determined by the global index k
            unsigned int global_tid = tid + blockDim.x * blockIdx.x;

            if ((global_tid & k) == 0) {
                // Sort ascending
                if (shared_data[tid] > shared_data[ixj]) {
                    // Swap elements in shared memory
                    short temp = shared_data[tid];
                    shared_data[tid] = shared_data[ixj];
                    shared_data[ixj] = temp;
                }
            } else {
                // Sort descending
                if (shared_data[tid] < shared_data[ixj]) {
                    // Swap elements in shared memory
                    short temp = shared_data[tid];
                    shared_data[tid] = shared_data[ixj];
                    shared_data[ixj] = temp;
                }
            }
        }
        // Synchronize to ensure all comparison-swaps for the current step are complete before starting the next step
        __syncthreads();
    }

    // Store data back to global memory and this is a coalesced write operation
    if (idx < n) {
        data[idx] = shared_data[tid];
    }
}

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));

    DTYPE* arrCpu = (DTYPE*)malloc(size * sizeof(DTYPE));

    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// arCpu contains the input random array
// arrSortedGpu should contain the sorted array copied from GPU to CPU
DTYPE *arrSortedGpu = (DTYPE*)malloc(size * sizeof(DTYPE));

// Transfer data (arr_cpu) to device

// Compute next power of 2 for padding
int paddedSize = 1;
while (paddedSize < size) {
    paddedSize <<= 1;
}

// Allocate device memory on the GPU for the padded array
DTYPE *d_arr;
cudaMalloc((void**)&d_arr, paddedSize * sizeof(DTYPE));

// Use pinned memory for faster transfers
DTYPE* paddedArray;
cudaMallocHost((void**)&paddedArray, paddedSize * sizeof(DTYPE));

// Prepare data in pinned memory and copy original data
memcpy(paddedArray, arrCpu, size * sizeof(DTYPE));

// Fill remaining padded space with sentinel values using std::fill
std::fill(paddedArray + size, paddedArray + paddedSize, SENTINEL);

// Single fast transfer to GPU
cudaMemcpy(d_arr, paddedArray, paddedSize * sizeof(DTYPE), cudaMemcpyHostToDevice);

// Free pinned memory
cudaFreeHost(paddedArray);

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    
/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// Perform bitonic sort on GPU 

// Define kernel launch parameters
int threadsPerBlock = 512; 
int blocks = (paddedSize + threadsPerBlock - 1) / threadsPerBlock;
int sharedMemBytes = threadsPerBlock * sizeof(DTYPE); 

// Sort the array in multiple passes
for (int k = 2; k <= paddedSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
        if (j >= threadsPerBlock) {
            // Sort using global memory for large intervals
            bitonic_Sort_Kernel<<<blocks, threadsPerBlock>>>(d_arr, j, k, paddedSize);
        } else {
            // Sort using shared memory for remaining small intervals and shared memory kernel handles all remaining j values in one call
            bitonic_Sort_Kernel_shared<<<blocks, threadsPerBlock, sharedMemBytes>>>(d_arr, j, k, paddedSize);
            break; // Exit inner loop since shared kernel handles all remaining j values
        }
    }
}

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start);

/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// Transfer sorted data back to host (copied to arrSortedGpu)

// Use pinned memory for faster return transfer
DTYPE* pinnedResult;
cudaMallocHost((void**)&pinnedResult, size * sizeof(DTYPE));

// Fast transfer from GPU to pinned memory
cudaMemcpy(pinnedResult, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

// Copy from pinned memory to final destination
memcpy(arrSortedGpu, pinnedResult, size * sizeof(DTYPE));

// Free pinned and device memories
cudaFreeHost(pinnedResult);
cudaFree(d_arr);

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // CPU sort for performance comparison
    std::sort(arrCpu, arrCpu + size);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;

    int match = 1;
    for (int i = 0; i < size; i++) {
        if (arrSortedGpu[i] != arrCpu[i]) {
            match = 0;
            break;
        }
    }

    free(arrCpu);
    free(arrSortedGpu);

    if (match)
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    else {
        printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
        return 0;
    }
    
    printf("\033[1;34mArray size         :\033[0m %d\n", size);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime/cpuTime) : (cpuTime/gpuTotalTime);
    float meps = size / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    return 0;
}