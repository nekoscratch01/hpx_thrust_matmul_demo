#include "matrix_utils.hpp"
#include "matmul.hpp"
#include <thrust/iterator/counting_iterator.h>
#include "hpx_thrust_bridge.hpp"

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/chrono.hpp>
#include <hpx/program_options.hpp>
#include <hpx/execution.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/async_cuda/cuda_executor.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>


// --- CUDA Error Check Macro ---
// This macro provides consistent error handling for CUDA operations
// It checks if CUDA operations succeeded and terminates HPX if they fail
#define CUDA_CHECK_MAIN(err)                                                  \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                             \
            std::cerr << "CUDA Main Error: " << cudaGetErrorString(err_)       \
                      << " in " << __FILE__ << ":" << __LINE__ << std::endl;   \
            try { hpx::terminate(); } catch(...) {}                             \
            throw std::runtime_error(std::string("CUDA critical error: ") + cudaGetErrorString(err_)); \
        }                                                                      \
    } while (0)

// Declaration of matrix multiplication CUDA kernel function defined in cuda_kernel_launcher.cu
extern void launch_matrix_multiplication(float* d_a, float* d_b, float* d_c, 
                                        std::size_t size, cudaStream_t stream);

// HPX main function
int hpx_main(hpx::program_options::variables_map& vm)
{
    // --- Setup & Matrix Generation/Loading ---
    std::size_t size = vm["size"].as<std::size_t>();
    if (size == 0) { 
        std::cerr << "Error: Matrix size cannot be zero." << std::endl;
        return hpx::finalize(); 
    }
    if (size > 8192) { 
        std::cerr << "Error: Matrix size is too large. Maximum allowed is 8192." << std::endl;
        return hpx::finalize(); 
    }
    
    std::cout << "Matrix size set to: " << size << "x" << size << std::endl;
    std::size_t num_elements = size * size;
    std::size_t matrix_bytes = num_elements * sizeof(float);
    std::string file_a = "matrix_a.txt";
    std::string file_b = "matrix_b.txt";
    
    try { 
        // Generate random matrices and save them to disk for reproducibility
        matrix_utils::generate_and_save_matrices(size, file_a, file_b); 
    } catch (const std::exception& e) { 
        std::cerr << "Error generating matrices: " << e.what() << std::endl;
        return hpx::finalize(); 
    }

    // Load matrices from files to ensure CPU and GPU use identical data
    std::vector<float> h_a, h_b;
    try {
        h_a = matrix_utils::load_matrix(file_a, size);
        h_b = matrix_utils::load_matrix(file_b, size);
    } catch (const std::exception& e) { 
        std::cerr << "Error loading matrices: " << e.what() << std::endl;
        return hpx::finalize(); 
    }
    
    std::vector<float> h_c_cpu(num_elements);
    std::vector<float> h_c_gpu(num_elements);

    // --- CPU Computation ---
    std::cout << "\n--- Starting CPU Computation (HPX Parallel) ---" << std::endl;
    hpx::chrono::high_resolution_timer timer_cpu;
    
    try { 
        multiply_cpu(h_a, h_b, h_c_cpu, size, hpx::execution::par); 
    } catch (const std::exception& e) { 
        std::cerr << "Error in CPU computation: " << e.what() << std::endl;
        return hpx::finalize(); 
    }
    
    double elapsed_cpu = timer_cpu.elapsed();
    std::cout << "CPU Time: " << elapsed_cpu << " seconds" << std::endl;
    matrix_utils::print_matrix("Result C (CPU, partial)", h_c_cpu, size);

    // --- GPU Computation ---
    std::cout << "\n--- Starting GPU Computation ---" << std::endl;
    hpx::chrono::high_resolution_timer timer_gpu_total;

    // Create CUDA stream for asynchronous operations
    cudaStream_t stream;
    CUDA_CHECK_MAIN(cudaStreamCreate(&stream));
    int device_id = 0;
    CUDA_CHECK_MAIN(cudaGetDevice(&device_id));
    hpx::cuda::experimental::cuda_executor gpu_executor(device_id);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    try {
        // 1. Allocate GPU memory for matrices
        // This reserves space on the GPU for our input and output data 
        CUDA_CHECK_MAIN(cudaMalloc(&d_a, matrix_bytes));
        CUDA_CHECK_MAIN(cudaMalloc(&d_b, matrix_bytes));
        CUDA_CHECK_MAIN(cudaMalloc(&d_c, matrix_bytes));
        std::cout << "[GPU] Device memory allocated." << std::endl;

        // 2. Copy input matrices A and B from host to device asynchronously
        // Using async operations allows overlapping transfers with computation
        std::cout << "[GPU] Posting H->D transfers..." << std::endl;
        CUDA_CHECK_MAIN(cudaMemcpyAsync(d_a, h_a.data(), matrix_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_MAIN(cudaMemcpyAsync(d_b, h_b.data(), matrix_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_MAIN(cudaStreamSynchronize(stream));
        std::cout << "[GPU] H->D transfers complete." << std::endl;

        // 3. Execute GPU matrix multiplication
        // Start a timer specifically for the kernel execution time
        std::cout << "[GPU] Starting kernel computation..." << std::endl;
        hpx::chrono::high_resolution_timer timer_gpu_kernel;
        
        // Call the CUDA kernel function defined in cuda_kernel_launcher.cu
        launch_matrix_multiplication(d_a, d_b, d_c, size, stream);
        CUDA_CHECK_MAIN(cudaStreamSynchronize(stream));
        
        double elapsed_gpu_kernel = timer_gpu_kernel.elapsed();
        std::cout << "[GPU] Kernel computation complete." << std::endl;

        // 4. Copy result matrix C from device back to host asynchronously
        std::cout << "[GPU] Posting D->H transfer for C..." << std::endl;
        CUDA_CHECK_MAIN(cudaMemcpyAsync(h_c_gpu.data(), d_c, matrix_bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_MAIN(cudaStreamSynchronize(stream));
        std::cout << "[GPU] D->H transfer complete." << std::endl;

        double elapsed_gpu_total = timer_gpu_total.elapsed();
        std::cout << "GPU Kernel Time: " << elapsed_gpu_kernel << " seconds" << std::endl;
        std::cout << "GPU Total Time (including transfers): " << elapsed_gpu_total << " seconds" << std::endl;
        matrix_utils::print_matrix("Result C (GPU, partial)", h_c_gpu, size);

        // 5. Free GPU memory resources
        std::cout << "[GPU] Freeing device memory..." << std::endl;
        CUDA_CHECK_MAIN(cudaFree(d_a)); d_a = nullptr;
        CUDA_CHECK_MAIN(cudaFree(d_b)); d_b = nullptr;
        CUDA_CHECK_MAIN(cudaFree(d_c)); d_c = nullptr;
        CUDA_CHECK_MAIN(cudaStreamDestroy(stream));
        std::cout << "[GPU] Device memory freed." << std::endl;

        // 6. Verify
        matrix_utils::verify_results(h_c_cpu, h_c_gpu, size);

    } catch (const hpx::exception& e) { 
        std::cerr << "HPX error in GPU computation: " << e.what() << std::endl;
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cudaStreamDestroy(stream);
        return hpx::finalize(); 
    } catch (const std::runtime_error& e) { 
        std::cerr << "Runtime error in GPU computation: " << e.what() << std::endl;
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cudaStreamDestroy(stream);
        return hpx::finalize(); 
    } catch (const std::exception& e) { 
        std::cerr << "Standard error in GPU computation: " << e.what() << std::endl;
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cudaStreamDestroy(stream);
        return hpx::finalize(); 
    } catch (...) { 
        std::cerr << "Unknown error in GPU computation" << std::endl;
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cudaStreamDestroy(stream);
        return hpx::finalize(); 
    }

    return hpx::finalize();
}

// Standard C++ main function
// Set up command line options
// Initialize HPX with our command line options
// Start HPX runtime and call hpx_main

int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_cmdline("Usage: hpx_matmul [options]");
    desc_cmdline.add_options()
        ("size", hpx::program_options::value<std::size_t>()->default_value(256),
         "Size N of the square matrices (N x N)");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_cmdline;

    return hpx::init(argc, argv, init_args);
}