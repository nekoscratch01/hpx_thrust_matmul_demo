cmake_minimum_required(VERSION 3.18 FATAL_ERROR) 
project(hpx_thrust_matmul_demo LANGUAGES CXX CUDA)

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "CMake Version: ${CMAKE_VERSION}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(HPX 1.9 REQUIRED COMPONENTS cuda)
message(STATUS "Found HPX: ${HPX_FOUND}")
message(STATUS "HPX Version: ${HPX_VERSION}")
message(STATUS "HPX Include Dirs: ${HPX_INCLUDE_DIRS}")
message(STATUS "HPX Libraries: ${HPX_LIBRARIES}")

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA: ${CUDA_FOUND}")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")

if(DEFINED HPX_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${HPX_CUDA_ARCHITECTURES})
  message(STATUS "Using CUDA Architectures from HPX: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(CMAKE_CUDA_ARCHITECTURES "80")
  message(WARNING "HPX_CUDA_ARCHITECTURES not found. Using default: ${CMAKE_CUDA_ARCHITECTURES}. Set manually if incorrect.")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

add_executable(hpx_matmul 
  src/main.cpp
  src/matmul_cpu.cpp
  src/cuda_kernel_launcher.cu
)

set_target_properties(hpx_matmul PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED}
    CXX_EXTENSIONS ${CMAKE_CXX_EXTENSIONS}
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    CUDA_STANDARD_REQUIRED ${CMAKE_CUDA_STANDARD_REQUIRED}
    CUDA_EXTENSIONS ${CMAKE_CUDA_EXTENSIONS}
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(hpx_matmul PRIVATE
    HPX::hpx
    CUDA::cudart
)

target_include_directories(hpx_matmul PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${HPX_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

message(STATUS "HPX Target Link Libraries: ${HPX_LIBRARIES}")
message(STATUS "CUDA Link Libraries: ${CUDA_LIBRARIES}")
message(STATUS "Configuring done. Use 'make' or your chosen generator to build.")