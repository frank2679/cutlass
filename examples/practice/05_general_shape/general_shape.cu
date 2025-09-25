#include <iostream>
#include <cute/tensor.hpp>
#include "gmem_desc_general_shape.hpp"
#include "utils.hpp"
#include "kernel.hpp"
#include <cutlass/half.h>
#include <cstdlib>
#include <ctime>

using namespace cute;
using half_t = cutlass::half_t;

int main() {
  // 获取MNK维度的值
  constexpr int M_VALUE = 128; // 128;
  constexpr int N_VALUE = 128; // 96;
  constexpr int K_VALUE = 64;
  
  // 分配设备内存
  half_t *d_a, *d_b, *d_c, *d_d;
  
  cudaMalloc(&d_a, sizeof(half_t) * M_VALUE * K_VALUE);
  cudaMalloc(&d_b, sizeof(half_t) * K_VALUE * N_VALUE);
  cudaMalloc(&d_c, sizeof(half_t) * M_VALUE * N_VALUE);
  cudaMalloc(&d_d, sizeof(half_t) * M_VALUE * N_VALUE);
  
  // 初始化数据
  half_t *h_a = new half_t[M_VALUE * K_VALUE];
  half_t *h_b = new half_t[K_VALUE * N_VALUE];
  half_t *h_c = new half_t[M_VALUE * N_VALUE];
  half_t *h_d = new half_t[M_VALUE * N_VALUE];
  half_t *h_d_cpu = new half_t[M_VALUE * N_VALUE]; // For CPU result
  
  // 初始化为随机值
  srand(time(NULL)); // 设置随机种子
  for (int i = 0; i < M_VALUE * K_VALUE; i++) h_a[i] = half_t(float(rand()) / float(RAND_MAX));// (1.0f); // (float(rand()) / float(RAND_MAX));
  for (int i = 0; i < K_VALUE * N_VALUE; i++) h_b[i] = half_t(float(rand()) / float(RAND_MAX));// (1.0f); // (float(rand()) / float(RAND_MAX));
  for (int i = 0; i < M_VALUE * N_VALUE; i++) h_c[i] = half_t(5.0f);
  
  cudaMemcpy(d_a, h_a, sizeof(half_t) * M_VALUE * K_VALUE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(half_t) * K_VALUE * N_VALUE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, sizeof(half_t) * M_VALUE * N_VALUE, cudaMemcpyHostToDevice);
  
  // 计算共享内存大小
  size_t shared_mem_size = sizeof(half_t) * (size(ATOM_M{}) * size(ATOM_K{}) + size(ATOM_K{}) * size(ATOM_N{}) + size(ATOM_M{}) * size(ATOM_N{}));
  std::cout << "Shared memory size: " << shared_mem_size << " bytes" << std::endl;

  // 启动测试内核
  test_gemm_with_dmem_desc_final<<<1, 1, shared_mem_size>>>(d_a, d_b, d_c, d_d, M_VALUE, N_VALUE, K_VALUE);
  
  // 同步并检查错误
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }
  
  // 拷贝结果回主机
  cudaMemcpy(h_d, d_d, sizeof(half_t) * M_VALUE * N_VALUE, cudaMemcpyDeviceToHost);
  
  // CPU GEMM计算
  cpu_gemm(h_a, h_b, h_c, h_d_cpu, M_VALUE, N_VALUE, K_VALUE);
  
  // 打印矩阵结果
  print_matrix(h_d_cpu, M_VALUE, N_VALUE, "CPU Result");
  print_matrix(h_d, M_VALUE, N_VALUE, "GPU Result");
  
  // 验证结果
  auto threshold = 0.05 * K_VALUE / size(ATOM_K{});
  bool verification_passed = compare_results(h_d, h_d_cpu, M_VALUE * N_VALUE, threshold);
  
  // 打印结果
  std::cout << "Result D value (GPU): " << float(h_d[0]) << std::endl;
  std::cout << "Result D value (CPU): " << float(h_d_cpu[0]) << std::endl;
  std::cout << "Verification given threshold " << threshold << ": " << (verification_passed ? "PASSED" : "FAILED") << std::endl;
  
  // 释放设备内存
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_d);
  
  // 释放主机内存
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  delete[] h_d;
  delete[] h_d_cpu;
  
  return 0;
}