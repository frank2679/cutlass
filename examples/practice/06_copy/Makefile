# Makefile for cute::copy example
# This example demonstrates using cute::copy to transfer data from global memory to shared memory

# Compiler and flags
NVCC = nvcc
ARCH = -arch=sm_90
INCLUDES = -I../../../include -I../../../tools/util/include
CXXFLAGS = -std=c++17 -O3 -diag-suppress 20013 -forward-unknown-to-host-compiler -Xcompiler -Wall

# Targets
TARGETS = copy_demo custom_copy_demo sm90_bulk_copy_demo

# Default target
all: $(TARGETS)

# Build rules
copy_demo: copy_demo.cu copy_kernel.cu
	$(NVCC) $(ARCH) $(INCLUDES) $(CXXFLAGS) -o $@ $^ -Xptxas -v

custom_copy_demo: custom_copy_main.cu custom_copy_kernel.cu copy_kernel.cu
	$(NVCC) $(ARCH) $(INCLUDES) $(CXXFLAGS) -o $@ $^ -Xptxas -v

sm90_bulk_copy_demo: sm90_bulk_copy_demo.cu
	$(NVCC) $(ARCH) $(INCLUDES) $(CXXFLAGS) -o $@ $^ -Xptxas -v

# Clean
clean:
	rm -f $(TARGETS)

.PHONY: all clean