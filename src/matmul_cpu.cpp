#include "matmul.hpp"
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_for_each.hpp>   // For hpx::for_each CPO
#include <hpx/include/parallel_execution.hpp> // For execution policies
#include <hpx/modules/iterator_support.hpp>   // Provides hpx::util::counting_iterator

#include <vector>
#include <cstddef> // For std::size_t
#include <iostream>
#include <stdexcept> // For invalid_argument

// Implementation of CPU matrix multiplication using HPX parallel algorithms
// This demonstrates how HPX can parallelize computation on the CPU
void multiply_cpu(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    std::size_t size,
    const hpx::execution::parallel_policy& policy)
{
    if (size == 0 || a.size() != size * size || b.size() != size * size || c.size() != size * size) {
         throw std::invalid_argument("CPU Matmul: Matrix size mismatch or zero size.");
    }
    std::cout << "[CPU] Starting matrix multiplication (" << size << "x" << size << ") using HPX "
              << typeid(policy).name() << " policy..." << std::endl;

    // Create counting iterators to iterate over all cells of the result matrix
    // These iterators are used instead of a traditional for loop for better parallelization
    auto begin = hpx::util::counting_iterator<std::size_t>(0);
    auto end = hpx::util::counting_iterator<std::size_t>(size * size);

    // Use HPX's parallel for_each to process all cells in parallel
    // This automatically distributes the work across available CPU cores
    hpx::for_each(policy,
        begin,
        end,
        [&](std::size_t idx) {
            // Convert 1D index to 2D coordinates (row, col)
            std::size_t row = idx / size;
            std::size_t col = idx % size;
            
            // Compute the dot product for this cell
            float sum = 0.0f;
            for (std::size_t k = 0; k < size; ++k) {
                sum += a[row * size + k] * b[k * size + col];
            }
            c[idx] = sum;
        });

     std::cout << "[CPU] Matrix multiplication finished." << std::endl;
}