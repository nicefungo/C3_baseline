#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>


// TODO
// Size is defined by N * K,
// Layout is to be decided  

#define CONV_WEIGHT_1_SIZE (1U << 9)
#define CONV_WEIGHT_2_SIZE (1U << 9)
#define CONV_WEIGHT_3_SIZE (1U << 10)
#define CONV_WEIGHT_m0_SIZE (1U << 8)
#define CONV_WEIGHT_m1_SIZE (9U << 8)

#define CONV_WEIGHT_1_OFFSET 0U
#define CONV_WEIGHT_2_OFFSET (CONV_WEIGHT_1_OFFSET + CONV_WEIGHT_1_SIZE)
#define CONV_WEIGHT_3_OFFSET (CONV_WEIGHT_2_OFFSET + CONV_WEIGHT_2_SIZE)
#define CONV_WEIGHT_m0_OFFSET (CONV_WEIGHT_3_OFFSET + CONV_WEIGHT_3_SIZE)
#define CONV_WEIGHT_m1_OFFSET (CONV_WEIGHT_m0_OFFSET + CONV_WEIGHT_m0_SIZE)

#define CONV_WEIGHT_SIZE 4608U

// Biases

#define CONV_BIAS_1_SIZE 16U
#define CONV_BIAS_2_SIZE 16U
#define CONV_BIAS_3_SIZE 32U
#define CONV_BIAS_m0_SIZE 16U
#define CONV_BIAS_m1_SIZE 16U


#define CONV_BIAS_1_OFFSET 0U
#define CONV_BIAS_2_OFFSET 16U
#define CONV_BIAS_3_OFFSET 32U
#define CONV_BIAS_m0_OFFSET 64U
#define CONV_BIAS_m1_OFFSET 80U

#define CONV_BIAS_SIZE 96U


// TODO
// input: N * C * W * H
// or N * C * H * W?

#define INPUT_SIZE (100U << 13)
#define OUTPUT_SIZE (100U << 13)


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
	    std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
	    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
	    std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
	    std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
	    std::cerr << cudaGetErrorString(err) << std::endl;
	    std::exit(EXIT_FAILURE);
    }
}

template<typename T>
__global__ void fused_25600x16x32_25600x16x16_SiLU(const T * input, const T * \
		Conv1_weight, const T * Conv1_bias, const T * Convm0_weight,\
	       	const T * Convm0_bias, T * D1, T * D2);

template<typename T>
__global__ void Convm1_trivial(const T * input, const T * weight, const T * \
		bias, T * D, unsigned int offset);

template<typename T>
__global__ void Convm1_25600x144x16_SiLU_adding(const T * input, const T * weight, \
		const T * bias, T * to_add, T * D);

// template<typename T>
// __global__ void fused_3200x16x16_3200x16x16_SiLU();


template<typename T>
__global__ void Conv_25600x16x32_SiLU(const T * input, const T * weight, \
		const T * bias, T * D);

template<typename T>
__global__ void Conv_25600x32x32_SiLU(const T * input, const T * weight, \
		const T * bias, T * D);

template<typename T>
void C3(const T * img, T * input, const T * weights, const T * biases, T * D, T * buffer1, T * buffer2, T * buffer3, T * reshaped_mat, T * reshaped_weight, T * out_img);

template<typename T, unsigned int TILING_MULTIPLY>
__global__ void im2col_32x160x160_25600x32_transpose(const T * img, T * D);

template<typename T, unsigned int TILING_MULTIPLY>
__global__ void col2im_25600x32_32x160x160_transpose(const T * D , T * img);

template<typename T, unsigned int TILING_MULTIPLY>
__global__ void col2im_25600x16_16x160x160_transpose(const T * D , T * img);

template<typename T>
__global__ void Convm1_weight_reshape_16x16x3x3_144x16(const T * weight , T * D);

template<typename T>
__global__ void Convm1_input_reshape_25600x16_25600x144(const T * weight , T * D);

template<typename T>
__global__ void concat_25600x16_25600x16_25600x32(const T * input1, const T * input2, T * D);
