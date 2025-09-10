#include <cuda_runtime.h>
#include <math.h>
#include <cuda_fp16.h>

template <typename T>
__host__ __device__ T gelu(const T x);

template <>
__host__ __device__ float gelu<float>(const float x) {
    const float sqrt2pi = sqrtf(2.0f / M_PI); 
    const float c = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(sqrt2pi * (x + c * x * x * x)));
}

template <>
__host__ __device__ __half gelu<__half>(const __half x) {
    const float sqrt2pi = sqrtf(2.0f / M_PI); 
    const float c = 0.044715f;
    
    float x_float = __half2float(x);  
    
    // Apply GELU formula in float and convert back to __half
    float result_float = 0.5f * x_float * (1.0f + tanhf(sqrt2pi * (x_float + c * x_float * x_float * x_float)));
    
    return __float2half(result_float);
}

template <typename T>
inline __host__ __device__ T silu(const T x);

template <>
inline __host__ __device__ float silu<float>(const float x) {
    return x / (1.0f + expf(-x));
}

template <>
inline __host__ __device__ __half silu<__half>(const __half x) {
    float x_float = __half2float(x);
    float result_float = x_float / (1.0f + expf(-x_float));
    return __float2half(result_float);
}

