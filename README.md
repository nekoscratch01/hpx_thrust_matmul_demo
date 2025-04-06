# HPX-Thrust Matrix Multiplication Example

## Project Core Functionality and Features

### 1. Asynchronous Execution Model
- Utilizes HPX's `cuda_executor` class to encapsulate CUDA operations and return `hpx::future` objects
- Allows users to use GPUs through the familiar HPX asynchronous programming model
- Implements asynchronous data transfer and kernel execution

### 2. Execution Strategy Bridging
- Creates custom execution strategy classes (`hpx::cuda::execution::par_policy`)
- Bridges HPX and Thrust execution models, enabling HPX algorithms to execute on GPUs
- Provides a unified interface for CPU and GPU execution

### 3. tag_invoke Mechanism
- Implements `tag_invoke` overloads, connecting HPX algorithm APIs with Thrust implementations
- Automatically routes algorithm calls to corresponding GPU implementations when specific execution strategies are used
- Maintains HPX API consistency while achieving GPU acceleration

## Implementation Details

### Matrix Multiplication Algorithm
The project implements standard matrix multiplication algorithm (C = A * B), with two versions:
1. **CPU Implementation**: Using HPX's parallel algorithms (`hpx::for_each`)
2. **GPU Implementation**: Using CUDA kernel functions, integrated into HPX through custom bridging mechanisms

### HPX-CUDA Integration
- Uses HPX's CUDA executor to handle asynchronous task submission and completion notification
- Uses CUDA streams to manage parallel execution on the GPU
- Provides type-safe and exception-safe interfaces

### Performance Comparison
In 1024×1024 matrix multiplication tests, performance comparison is as follows:
- CPU computation time: approximately 0.121 seconds
- GPU kernel execution time: approximately 0.001 seconds
- GPU total time (including data transfer): approximately 0.170 seconds

## Project Structure

```
hpx_thrust_matmul_demo/
├── src/
│   ├── main.cpp                // Main program: matrix generation, CPU/GPU computation calls, result verification
│   ├── matmul_cpu.cpp          // CPU matrix multiplication implementation
│   ├── matmul.hpp              // Matrix multiplication declarations
│   ├── matrix_utils.hpp        // Matrix utility functions (generation, loading, verification)
│   ├── hpx_thrust_bridge.hpp   // HPX-Thrust bridge header file: custom execution strategies and tag_invoke overloads
│   └── cuda_kernel_launcher.cu // CUDA kernel implementation: matrix multiplication GPU implementation and kernel launch functions
└── CMakeLists.txt              // Build system configuration
```

## Building and Running

### Prerequisites
- CMake 3.18+
- C++17 compatible compiler
- CUDA 11.0+
- HPX 1.9+ (compiled with CUDA components)
- NVIDIA GPU with CUDA support

### Build Steps
```bash
mkdir build && cd build
cmake .. \
    -DHPX_DIR=/path/to/hpx/lib/cmake/HPX \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES='80'  # Adjust according to GPU type, e.g., 70(V100), 80(A100), 86(RTX 3090), etc.
make -j $(nproc)
```

### Running the Example
```bash
./hpx_matmul --size 1024  # Specify matrix size (N×N)
```

## Possible Extensions

- Add GPU implementations for other HPX parallel algorithms
- Optimize GPU kernel implementation, such as using shared memory or block-level matrix multiplication
- Support non-square and sparse matrix calculations
- Add support for other accelerators (such as AMD ROCm)