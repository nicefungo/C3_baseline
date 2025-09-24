#include <cuda_runtime.h>
#include <math.h>
#include <cuda_fp16.h>
#include "cnpy.h"
#include <cstring>
#include <stdexcept>
#include <string>
#include <fstream>

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

// FIXED FOR BASELINE
template <class T>
void load_npy_into(const std::string& path, T* dst, size_t expected_count) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    const size_t elem_bytes = arr.word_size;
    const size_t count = arr.num_vals;
    if (elem_bytes != sizeof(T))
        throw std::runtime_error("dtype/word_size mismatch: file elem size = " +
                                 std::to_string(elem_bytes) + ", T = " +
                                 std::to_string(sizeof(T)));
    if (expected_count && count != expected_count)
        throw std::runtime_error("element count mismatch: file=" +
                                 std::to_string(count) + " expected=" +
                                 std::to_string(expected_count));
    // arr.data<T>() is owned by 'arr'; copy it out while 'arr' is alive
    std::memcpy(dst, arr.data<T>(), count * sizeof(T));
}


// FIXED FOR BASELINE
template <class T>
void load_input_into(const std::string& path, T * dst, size_t expected_count){
	std::ifstream in(path);
	if(!in){
		std::cerr << "Input file " << path << " doesn't exist!" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	size_t i = 0;
	for(; i < expected_count && in >> dst[i]; ++i){}

}
