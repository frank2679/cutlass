# Makefile for GEMM with gmemDesc example
# This example demonstrates using gmemDesc with cute::gemm()

# Compiler and flags
NVCC = nvcc
ARCH = -arch=sm_80
INCLUDES = -I../../../include -I../../../tools/util/include
CXXFLAGS = -std=c++17 -O3 -diag-suppress 20013 -forward-unknown-to-host-compiler -Xcompiler -Wall

# Targets
TARGETS = general_shape

# Default target
all: $(TARGETS)

# Build rules
general_shape: general_shape.cu gmem_desc_general_shape.hpp utils.hpp kernel.hpp
	$(NVCC) $(ARCH) $(INCLUDES) $(CXXFLAGS) -o $@ $< -Xptxas -v

# Clean
clean:
	rm -f $(TARGETS)

.PHONY: all clean