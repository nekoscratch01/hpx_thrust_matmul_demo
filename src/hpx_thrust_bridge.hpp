// src/hpx_thrust_bridge.hpp
// HPX-Thrust bridge header: Defines custom execution policy and tag_invoke overloads

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/execution.hpp>         // HPX execution policies and CPOs
#include <hpx/execution/executors/execution.hpp> // for hpx::parallel::execution::post
#include <hpx/async_cuda/cuda_executor.hpp>    // HPX CUDA executor (experimental)
#include <hpx/futures/future.hpp>              // hpx::future
#include <hpx/parallel/algorithms/for_each.hpp>// For hpx::for_each_t CPO tag

// Thrust includes
#include <thrust/execution_policy.h>           // For thrust::cuda::par
#include <thrust/system/cuda/execution_policy.h> // For .on()
#include <thrust/for_each.h>                   // The Thrust algorithm we bridge
#include <thrust/system/system_error.h>        // For catching thrust exceptions

#include <cuda_runtime.h> // For cudaStream_t, cudaGetDevice, etc.

#include <iostream>
#include <stdexcept>     // For exceptions
#include <string>        // For error messages
#include <type_traits>   // For std::forward, std::decay_t
#include <utility>       // For std::move

// --- CUDA Error Check Macro ---
// Provides standardized error checking for CUDA operations in bridge code
#define CUDA_CHECK_BRIDGE(err)                                                \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA Bridge Error: ") +       \
                                     cudaGetErrorString(err_) +                 \
                                     " in " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while (0)


namespace hpx { namespace cuda { namespace execution {

    //
    // Custom CUDA execution policy class
    // This is the key bridge between HPX algorithms and Thrust GPU implementations
    //
    class par_policy {
    private:
        cudaStream_t stream_{0};  // CUDA stream to use for operations
        int device_id_{0};        // CUDA device ID to use

        // Helper to get the current CUDA device
        // This is called during policy construction to capture the active device
        static int get_current_device() {
            int current_device = 0;
            // FIX: Correct character encoding error
            cudaError_t err = cudaGetDevice(&current_device);
            if (err != cudaSuccess) {
                 std::cerr << "Warning: cudaGetDevice failed in par_policy constructor. Assuming device 0. Error: "
                           << cudaGetErrorString(err) << std::endl;
                 current_device = 0;
            }
            return current_device;
        }

    public:
        // Default constructor - uses default stream on current device
        par_policy() noexcept : device_id_(get_current_device()) {}
        
        // Stream constructor - uses specified stream on current device
        explicit par_policy(cudaStream_t stream) noexcept
            : stream_(stream), device_id_(get_current_device()) {}
            
        // Full constructor - uses specified device and stream
        explicit par_policy(int device_id, cudaStream_t stream = 0) noexcept
             : stream_(stream), device_id_(device_id) {}

        // Accessor methods to get internal state
        [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
        [[nodiscard]] int device() const noexcept { return device_id_; }
        
        // Creates and returns a Thrust execution policy using our stream
        // This is a key method that bridges HPX and Thrust worlds
        [[nodiscard]] auto thrust_policy() const { return thrust::cuda::par.on(stream_); }
        
        // Comparison operators for policy objects
        bool operator==(const par_policy& other) const noexcept {
             return stream_ == other.stream_ && device_id_ == other.device_id_;
        }
        bool operator!=(const par_policy& other) const noexcept { return !(*this == other); }
    };

    // Global instance - users can directly use: hpx::cuda::execution::par
    inline par_policy par{};

    //
    // tag_invoke overload for hpx::for_each
    // This is the core implementation of the HPX-Thrust bridge pattern
    // When HPX for_each is called with our CUDA policy, this function intercepts the call
    //
    template <typename Iterator, typename Func>
    hpx::future<void>
    tag_invoke(hpx::for_each_t,               // Tag type representing the for_each algorithm
               const par_policy& policy,      // Our custom CUDA execution policy
               Iterator first, Iterator last, // Iterator range
               Func&& f)                      // Function to apply
    {
        int device_id = policy.device();
        // Create HPX CUDA executor to handle async GPU operations
        // This is a lightweight object that manages GPU task execution
        hpx::cuda::experimental::cuda_executor executor(device_id);

        std::cout << "[HPX Bridge] tag_invoke for hpx::for_each triggered (GPU path)."
                  << " Device: " << device_id << ", Policy Stream: " << policy.stream() << std::endl;

        // Store the function in a shared_ptr to ensure it stays alive through async execution
        // This is important since we're passing it to async operations
        auto func_ptr = std::make_shared<typename std::decay<Func>::type>(std::forward<Func>(f));
        
        // Create a lambda that will execute on the GPU
        // This uses a simple function signature for better cuda_executor compatibility
        auto thrust_lambda = [policy, first, last, func_ptr]() {
            try {
                std::cout << "[HPX Bridge] Executing thrust::for_each in posted lambda..." << std::endl;
                
                // Get the Thrust execution policy and call the actual Thrust implementation
                auto thrust_policy = policy.thrust_policy();
                thrust::for_each(thrust_policy, first, last, *func_ptr);
                
                std::cout << "[HPX Bridge] thrust::for_each submitted/completed." << std::endl;
            } catch (const thrust::system_error& e) {
                // Catch and handle Thrust-specific errors
                std::cerr << "[HPX Bridge] Thrust Error caught: " << e.what() << std::endl;
                throw std::runtime_error(std::string("Thrust execution failed: ") + e.what());
            } catch (const std::exception& e) {
                // Handle standard errors
                std::cerr << "[HPX Bridge] Standard Error caught: " << e.what() << std::endl;
                throw;
            } catch (...) {
                // Catch-all for any other errors
                std::cerr << "[HPX Bridge] Unknown error caught." << std::endl;
                throw std::runtime_error("Unknown error during Thrust execution via HPX Bridge.");
            }
        };

        // Post the lambda to the executor
        // This schedules the GPU work asynchronously
        hpx::parallel::execution::post(executor, thrust_lambda);
        
        std::cout << "[HPX Bridge] Posted task to executor. Returning future." << std::endl;
        
        // Return a future that will become ready when the GPU work completes
        return executor.get_future();
    }

}}} // namespace hpx::cuda::execution