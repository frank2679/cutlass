# 06_copy - CUTE Copy Demo

This demo showcases how to use `cute::copy` to transfer data between global memory and shared memory in CUDA using the CUTE (CUDA Template Extensions) library.

## Purpose

The main goal of this demo is to demonstrate:
1. How to use CUTE tensors for memory management
2. How to copy data from global memory to shared memory using `cute::copy`
3. How to copy data from shared memory back to global memory
4. Proper synchronization techniques when working with shared memory

## Key Concepts

- **Global Memory**: Device memory accessible by all threads
- **Shared Memory**: On-chip memory shared among threads in a block
- **CUTE Tensors**: Multi-dimensional data structures with layout information
- **Tiling**: Dividing large matrices into smaller tiles for efficient processing
- **Synchronization**: Using `__syncthreads()` to coordinate thread access to shared memory

## Code Structure

The code is organized into the following files:
- `copy_demo.cu`: Main CUDA file with kernel implementation
- `utils.hpp`: Utility functions for matrix initialization and verification
- `Makefile`: Build configuration

## How It Works

1. The demo creates two 64x64 matrices in global memory
2. It divides the matrices into 32x32 tiles
3. For each tile:
   - Copy data from global memory to shared memory using `cute::copy`
   - Copy data from shared memory to another shared memory buffer
   - Copy data from shared memory back to global memory
4. The result is verified by comparing the output with the input

## Building and Running

```bash
# Build the demo
make

# Run the demo
./copy_demo
```

## Expected Output

The demo should show that all elements match within a small epsilon, indicating successful data transfer between global and shared memory.