#pragma once

#include <vector>
#include <hpx/hpx.hpp> // Include HPX core for execution policies

// Function declaration for CPU matrix multiplication implementation
// This function computes the product of matrices A and B on the CPU using HPX parallelism
// Parameters:
//   a - Input matrix A stored as a flattened vector
//   b - Input matrix B stored as a flattened vector 
//   c - Output matrix C (will be overwritten with the result)
//   size - Dimension of the square matrices (N for N×N matrices)
//   policy - HPX execution policy that controls parallelization strategy
void multiply_cpu(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    std::size_t size,
    const hpx::execution::parallel_policy& policy = hpx::execution::par
);