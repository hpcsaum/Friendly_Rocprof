/* HIP backend. Compiled by hipcc regardless of COMPILER, and given a C
 * linkage entry point so main.c can declare and call it directly. */
#include <hip/hip_runtime.h>

__global__ void stencil_kernel(double *array, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < n - 1) {
        array[i] = 0.5 * (array[i - 1] + array[i + 1]);
    }
}

extern "C" void launch_hip_kernel(double *host_array, int n) {
    double *d_array;
    size_t bytes = (size_t)n * sizeof(double);
    hipMalloc(&d_array, bytes);
    hipMemcpy(d_array, host_array, bytes, hipMemcpyHostToDevice);

    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(stencil_kernel, dim3(blocks), dim3(threads), 0, 0, d_array, n);
    hipDeviceSynchronize();

    hipMemcpy(host_array, d_array, bytes, hipMemcpyDeviceToHost);
    hipFree(d_array);
}
