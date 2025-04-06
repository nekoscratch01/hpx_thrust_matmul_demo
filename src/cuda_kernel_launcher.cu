// CUDA kernel implementation for matrix multiplication and kernel launch function

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>

// CUDA kernel function for matrix multiplication
// This code will be executed on the GPU device
// __global__ indicates this is a CUDA kernel function that can be called from the host
__global__ void matmul_kernel(const float* a, const float* b, float* c, std::size_t size) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size * size) return;
    
    std::size_t row = idx / size;
    std::size_t col = idx % size;
    float sum = 0.0f;
    for (std::size_t k = 0; k < size; ++k) {
        sum += a[row * size + k] * b[k * size + col];
    }
    c[idx] = sum;
}

// Host function to launch the matrix multiplication kernel
// This is a C++ function that can be called from HPX
void launch_matrix_multiplication(float* d_a, float* d_b, float* d_c, 
                                std::size_t size, cudaStream_t stream) {
    std::cout << "[CUDA] Launching matrix multiplication kernel..." << std::endl;
    
    // Calculate kernel launch configuration
    // block_size: number of threads per block (CUDA thread block)
    // grid_size: number of blocks to launch (calculated to cover all elements)
    const int block_size = 256;
    const int grid_size = (size * size + block_size - 1) / block_size;
    
    // Launch the CUDA kernel with the specified configuration
    // The <<< >>> syntax is specific to CUDA and specifies the execution configuration
    // Parameters: grid_size (blocks), block_size (threads per block), shared_memory, stream
    matmul_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_c, size);
    
    // Check for kernel launch errors
    // Note: This only checks if the kernel was launched successfully,
    // not if it executed successfully
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch error: ") + 
                                 cudaGetErrorString(err));
    }
    
    std::cout << "[CUDA] Kernel launched successfully." << std::endl;
}

// Alternative implementation using Thrust
// This is exposed as a C function for easier integration with other languages
// This implementation is simplified to use direct CUDA calls instead of Thrust algorithms
extern "C" void execute_thrust_for_each(void* a_ptr, void* b_ptr, void* c_ptr, 
                                      std::size_t size) {
    try {
        std::cout << "[Thrust] Executing alternative implementation..." << std::endl;
        
        float* d_a = static_cast<float*>(a_ptr);
        float* d_b = static_cast<float*>(b_ptr);
        float* d_c = static_cast<float*>(c_ptr);
        // Use the default CUDA stream (0)
        cudaStream_t stream = 0;
        
        // Launch the same CUDA kernel as above but from this function
        // This could have been implemented using thrust::for_each instead
        const int block_size = 256;
        const int grid_size = (size * size + block_size - 1) / block_size;
        matmul_kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_c, size);
        
        // Synchronize to ensure completion before returning
        // This makes the function call blocking (synchronous)
        cudaStreamSynchronize(stream);
        
        std::cout << "[Thrust] Execution complete." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Thrust] Error: " << e.what() << std::endl;
        throw;
    }
}