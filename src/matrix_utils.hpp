#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <stdexcept>
#include <iostream>
#include <iomanip>    // For std::setw, std::fixed, std::setprecision
#include <cmath>      // For std::fabs
#include <limits>     // For numeric_limits
#include <algorithm>  // For std::min
#include <sstream>    // For robust file reading

namespace matrix_utils {

// Generates two size x size matrices with random float values and saves them
inline void generate_and_save_matrices(std::size_t size,
                                       const std::string& file_a,
                                       const std::string& file_b) {
    if (size == 0) throw std::invalid_argument("Matrix size cannot be zero.");
    if (size > 10000) std::cerr << "Warning: Generating very large matrices..." << std::endl;

    std::vector<float> a(size * size);
    std::vector<float> b(size * size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::cout << "Generating matrix A (" << size << "x" << size << ") to " << file_a << "..." << std::endl;
    std::ofstream ofs_a(file_a);
    if (!ofs_a) throw std::runtime_error("Cannot open file for writing: " + file_a);
    ofs_a << std::fixed << std::setprecision(8);
    for (std::size_t i = 0; i < size; ++i) {
        for (std::size_t j = 0; j < size; ++j) {
            a[i * size + j] = dis(gen);
            ofs_a << a[i * size + j] << (j == size - 1 ? "" : " ");
        }
        ofs_a << "\n";
    }
    ofs_a.close(); std::cout << "Matrix A generated." << std::endl;

    std::cout << "Generating matrix B (" << size << "x" << size << ") to " << file_b << "..." << std::endl;
    std::ofstream ofs_b(file_b);
    if (!ofs_b) throw std::runtime_error("Cannot open file for writing: " + file_b);
    ofs_b << std::fixed << std::setprecision(8);
    for (std::size_t i = 0; i < size; ++i) {
        for (std::size_t j = 0; j < size; ++j) {
            b[i * size + j] = dis(gen);
            ofs_b << b[i * size + j] << (j == size - 1 ? "" : " ");
        }
         ofs_b << "\n";
    }
    ofs_b.close(); std::cout << "Matrix B generated." << std::endl;
}

// Loads a matrix from a text file (space-separated, row per line)
inline std::vector<float> load_matrix(const std::string& filename, std::size_t expected_size) {
    if (expected_size == 0) throw std::invalid_argument("Expected matrix size cannot be zero.");
    std::cout << "Loading matrix from " << filename << " (expecting " << expected_size << "x" << expected_size << ")..." << std::endl;
    std::vector<float> matrix_data;
    matrix_data.reserve(expected_size * expected_size);
    std::ifstream ifs(filename);
    if (!ifs) throw std::runtime_error("Cannot open file for reading: " + filename);

    float val;
    std::size_t count = 0;
    std::string line;
    std::size_t line_num = 0;
    while (std::getline(ifs, line)) {
        line_num++;
        std::stringstream ss(line);
        std::size_t col_count = 0;
        while (ss >> val) {
            if (count >= expected_size * expected_size) throw std::runtime_error("Matrix file " + filename + " contains more than expected elements...");
            matrix_data.push_back(val);
            count++;
            col_count++;
        }
        if (!ss.eof() && ss.fail() && !ss.bad()) throw std::runtime_error("Error reading numeric value from file " + filename + " at line " + std::to_string(line_num));
        if (col_count != 0 && col_count != expected_size && line_num <= expected_size) throw std::runtime_error("Matrix file " + filename + " has incorrect columns at line " + std::to_string(line_num) + "...");
    }
    if (count != expected_size * expected_size) throw std::runtime_error("Matrix file " + filename + " does not contain expected number of elements...");
    std::cout << "Matrix loaded successfully (" << expected_size << "x" << expected_size << ")." << std::endl;
    return matrix_data;
}

// Prints a small matrix (useful for debugging)
inline void print_matrix(const std::string& title, const std::vector<float>& matrix, std::size_t size, std::size_t limit = 5) {
    std::cout << "--- " << title << " ---" << std::endl;
    if (matrix.empty() || size == 0 || matrix.size() != size * size) { /* ... error message ... */ return; }
    std::size_t print_limit = std::min(limit, size);
    std::cout << std::fixed << std::setprecision(3);
    for (std::size_t i = 0; i < print_limit; ++i) {
        std::cout << "  ";
        for (std::size_t j = 0; j < print_limit; ++j) { std::cout << std::setw(8) << matrix[i * size + j] << " "; }
        if (print_limit < size) std::cout << "...";
        std::cout << std::endl;
    }
     if (print_limit < size) { /* ... print ellipsis row ... */ }
    std::cout << "---------------------" << std::endl;
}

// Verifies if two matrices are approximately equal
inline bool verify_results(const std::vector<float>& mat_ref, const std::vector<float>& mat_check, std::size_t size, float abs_tol = 1e-4f, float rel_tol = 1e-3f) {
    if (size == 0 || mat_ref.size() != size * size || mat_check.size() != size * size) { /* ... error message ... */ return false; }
    std::cout << "Verifying results (abs_tol=" << abs_tol << ", rel_tol=" << rel_tol << ")..." << std::endl;
    std::size_t errors = 0;
    std::size_t max_errors_to_print = 10;
    for (std::size_t i = 0; i < size * size; ++i) {
        float ref_val = mat_ref[i]; float check_val = mat_check[i]; float diff = std::fabs(ref_val - check_val);
        if (diff > std::max(abs_tol, rel_tol * std::max(std::fabs(ref_val), std::fabs(check_val)))) {
            errors++;
            if (errors <= max_errors_to_print) { /* ... print error details ... */ }
            if (errors == max_errors_to_print + 1) { /* ... print suppression message ... */ }
        }
    }
    if (errors == 0) { std::cout << "Verification successful!" << std::endl; return true; }
    else { std::cout << "Verification failed with " << errors << " differing elements." << std::endl; return false; }
}

} // namespace matrix_utils