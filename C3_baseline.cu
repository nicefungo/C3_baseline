#include "C3_baseline.h"
#include "util.h"

void TODO(){
	std::cout << "TODO stuff undone in `cuda_baseline.cu`." << std::endl;
	std::exit(EXIT_FAILURE);
}


template<typename T>
__global__ void im2col_32x160x160_25600x32_transpose(const T * img, T * D){
	

	// TILE_DIM = 32
	__shared__ T tile[32][33]; // tile size is 1 * 800

	// unsigned int thread_linear = threadIdx.y * blockDim.x + threadIdx.x; // 0:1024
	// unsigned int block_linear = blockIdx.y * gridDim.x + blockIdx.x; // 0:800

	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; // 0:25600
	unsigned int col = threadIdx.y; // 0:32

	// Let a single block transposes one tile (1024 elements)
	// Requires (800, 1) blocks
	if(col >= 32 || row >= 25600) return;

	// One thread moves 1024/BLOCK_DIM^2 elements
	// Assume a typical (32, 32) blocksize, then it is 1
	
	tile[threadIdx.y][threadIdx.x] = img[col * 25600 + row];
	
	__syncthreads();

	D[(((blockIdx.x << 5) + threadIdx.y) << 5) + threadIdx.x] = tile[threadIdx.x][threadIdx.y];


}

template<typename T>
__global__ void col2im_25600x32_32x160x160_transpose(const T * D , T * img){
	
	__shared__ T tile[32][33];

	// unsigned int thread_linear = threadIdx.y * blockDim.x + threadIdx.x; // 0:1024
        // unsigned int block_linear = blockIdx.y * gridDim.x + blockIdx.x; // 0:800

        unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; // 0:25600
        unsigned int col = threadIdx.y; // 0:32

	if(col >= 32 || row >= 25600) return;
	
	tile[threadIdx.y][threadIdx.x] = D[(((blockIdx.x << 5) + threadIdx.y) << 5) +threadIdx.x];

	__syncthreads();
        
	img[(col * 25600) + row] = tile[threadIdx.x][threadIdx.y];

}


template<typename T>
__global__ void fused_25600x16x32_25600x16x16_SiLU(const T * input, const T * \
                Conv1_weight, const T * Conv1_bias, const T * Convm0_weight,\
                const T * Convm0_bias, T * D1, T * D2)
{
	//
	// Probably won't need extra global memory
	// Shared A_tile is of size 32 * 32
	
	__shared__ T A_tile[32][32];
	__shared__ T B_tile[32][16];

	// For temporary 
	// __shared__ T D_tile[32][16];

	// unsigned int start_pos = (offset * 3200U) << 5;
	unsigned int tile_start_pos = (blockIdx.y << 10);

	// Typical block size dim3(16, 16)
	// Each thread move 4 elements from input,
	// 2 elements from Conv1_weight and 1 element from Conv2_weigt
	// Each thread computes 2 elements in D1 and D2

	unsigned int block_linear = gridDim.x * blockIdx.y + blockIdx.x;
        unsigned int thread_linear = blockDim.x * threadIdx.y + threadIdx.x;	

	unsigned int warp_id = thread_linear >> 5;
	unsigned int lane_id = thread_linear & 31U;

	A_tile[warp_id][lane_id] = input[tile_start_pos + thread_linear];
	A_tile[warp_id + 8][lane_id] = input[tile_start_pos + thread_linear + 256];
	A_tile[warp_id + 16][lane_id] = input[tile_start_pos + thread_linear + 512];
	A_tile[warp_id + 24][lane_id] = input[tile_start_pos + thread_linear + 768];

	B_tile[threadIdx.y][threadIdx.x] = Conv1_weight[(threadIdx.y << 4) + threadIdx.x];
	B_tile[threadIdx.y + 16][threadIdx.x] = Conv1_weight[(threadIdx.y << 4) + threadIdx.x + 256];
       
	T sum0{0};
	T sum1{0};

	T B_val[32];

	__syncthreads();

#pragma unroll
	for(int i = 0; i < 32; i++){
		B_val[i] = B_tile[i][threadIdx.x];
	}

#pragma unroll
	for(int i = 0; i < 32; i++){
		sum0 += A_tile[threadIdx.y][i] * B_val[i];
		sum1 += A_tile[threadIdx.y + 16][i] * B_val[i];
	}	

	sum0 += Conv1_bias[threadIdx.x];
	sum1 += Conv1_bias[threadIdx.x];
	
	sum0 = silu<T>(sum0);
	sum1 = silu<T>(sum1);

	// to global
	D1[(tile_start_pos >> 1) + (threadIdx.y << 4) + threadIdx.x] = sum0;
	D1[(tile_start_pos >> 1) + (threadIdx.y << 4) + threadIdx.x + 256] = sum1;
	
	// to shared
	A_tile[threadIdx.y][threadIdx.x] = sum0;
	A_tile[threadIdx.y + 16][threadIdx.x] = sum1;

	B_tile[threadIdx.y][threadIdx.x] = Convm0_weight[(threadIdx.y << 4)
						        + threadIdx.x];
	

	sum0 = static_cast<T>(0);
	sum1 = static_cast<T>(0);
	
	__syncthreads();

#pragma unroll
	for(int i = 0; i < 16; i++){
		B_val[i] = B_tile[i][threadIdx.x];
	}

#pragma unroll	
	for(int i = 0; i < 16; i++){
		sum0 += A_tile[threadIdx.y][i] * B_val[i];
		sum1 += A_tile[threadIdx.y + 16][i] * B_val[i];
	}
	
	sum0 += Convm0_bias[threadIdx.x];
	sum1 += Convm0_bias[threadIdx.x];

	sum0 = silu<T>(sum0);
	sum1 = silu<T>(sum1);

	D2[(tile_start_pos >> 1) + (threadIdx.y << 4) + threadIdx.x] = sum0; 
		// + input[(tile_start_pos >> 1) + (threadIdx.y << 4) + threadIdx.x];
        D2[(tile_start_pos >> 1) + (threadIdx.y << 4) + threadIdx.x + 256] = sum1;
		// + input[(tile_start_pos >> 1) + (threadIdx.y << 4) + threadIdx.x + 256];

	return;
}


void CPU_Convm1_25600x16_16x16x3x3(const float* X, const float* Wt, const float* bias, float* Y) {
    for (int h = 0; h < 160; ++h) {
        for (int w = 0; w < 160; ++w) {
            const int r = h * 160 + w;
            for (int m = 0; m < 16; ++m) {
                float acc = bias ? bias[m] : 0.0f;
                for (int c = 0; c < 16; ++c) {
                    for (int kh = -1; kh <= 1; ++kh) {
                        const int ih = h + kh;
                        if (ih < 0 || ih >= 160) continue;
                        for (int kw = -1; kw <= 1; ++kw) {
                            const int iw = w + kw;
                            if (iw < 0 || iw >= 160) continue;
                            const int slot = (kh + 1) * 3 + (kw + 1);      // 0..8
                            const float x = X[((ih * 160 + iw) * 16) + c];
                            const float wv = Wt[(((m * 16 + c) * 3 + (kh + 1)) * 3) + (kw + 1)];
                            acc += x * wv;
                        }
                    }
                }
		
                Y[r * 16 + m] = silu<float>(acc);
            }
        }
    }
}

template<typename T>
__global__ void Convm1_trivial(const T * input, const T * weight, const T * \
                bias, T * D, unsigned int offset)
{
	// Trivial convolution
	// consider (9 * 25600) * 32 input
	// 
	// each offset compute (9 *3200) * 32 in output
	// 
	// 3x3 kernel size, (1, 1) padding 
	//
	TODO();

}


template<typename T>
__global__ void Convm1_25600x144x16_SiLU_adding(const T * input, const T * weight, \
                const T * bias, T * to_add, T * D)
{
	// tile (32, 16)
	// grid(800, 1) block(16, 16) split_k 9
	
	__shared__ T tile_A[32][17];
	__shared__ T tile_B[16][17];

	
	unsigned int thread_linear = blockDim.x * threadIdx.y + threadIdx.x;

	T sum0{0};
	T sum1{0};

#pragma unroll
	for(int i = 0; i < 9; i++){
		// i k_axis_offset

		tile_A[threadIdx.y][threadIdx.x] = input[((blockIdx.x << 5) + threadIdx.y) * 144 + (i << 4) + threadIdx.x];
		tile_A[threadIdx.y + 16][threadIdx.x] = input[((blockIdx.x << 5) + threadIdx.y + 16U) * 144 + (i << 4) + threadIdx.x];

		tile_B[threadIdx.y][threadIdx.x] = weight[(i << 8) + thread_linear];

		__syncthreads();

		T B_val[16];

#pragma unroll
		for(int j = 0; j < 16; j++){
			B_val[j] = tile_B[j][threadIdx.x];	
		}

#pragma unroll
		for(int j = 0; j < 16; j++){
		
			sum0 += tile_A[threadIdx.y][j] * B_val[j];	
			sum1 += tile_A[threadIdx.y + 16][j] * B_val[j];	
		}

		__syncthreads();

	}

	T the_bias = bias[threadIdx.x];
	sum0 += the_bias;
	sum1 += the_bias;

	sum0 = silu<T>(sum0);
	sum1 = silu<T>(sum1);

	if(to_add){
		sum0 += to_add[(((blockIdx.x << 5) + threadIdx.y) << 4) + threadIdx.x];
		sum1 += to_add[(((blockIdx.x << 5) + threadIdx.y + 16U) << 4) + threadIdx.x];
	}

	D[(((blockIdx.x << 5) + threadIdx.y) << 4) + threadIdx.x] = sum0;
	D[(((blockIdx.x << 5) + threadIdx.y + 16U) << 4) + threadIdx.x] = sum1;

}

template<typename T>
__global__ void Convm1_weight_reshape_16x16x3x3_144x16(const T * weight , T * D)
{

	
	// (9, 1) grid, (16, 16) block

	__shared__ T tile[16][17];

	
	// unsigned int thread_linear = blockDim.x * threadIdx.y + threadIdx.x;
	unsigned int row = threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	

	tile[threadIdx.y][threadIdx.x] = weight[row * 144 + col];
	
	__syncthreads();

	D[(blockIdx.x * 16 + threadIdx.y) * 16 + threadIdx.x] = tile[threadIdx.x][threadIdx.y];

	return;	
}


// mem intensive
template<typename T>
__global__ void Convm1_input_reshape_25600x16_25600x144(const T * input , T * D)
{
	// Each output tile is 160x(16x9)
	// Necessary gridsize is (160, 1)
	// 

	// int col = blockIdx.x * blockDim.x + threadIdx.x;
	// int row = blockIdx.y * blockDim.y + threadIdx.y;

	// Assume (32, 8) block size
	// TODO: try (32, 16)
	// Receiptive region size is (160x3)x16

	// tile start row number
	unsigned int tile_row = blockIdx.x * 160U; // 0:25600:160
	// unsigned int tile_col = 0;

	// mid-of-kernel element pos
	unsigned int input_tile_start_pos = (tile_row << 4);
	unsigned int output_tile_start_pos = tile_row * 144;
	
	unsigned int thread_linear = threadIdx.y * blockDim.x + threadIdx.x;  // 256 for (32, 8)
	// unsigned int input_offset = 0;
	// unsigned int output_offset = 0;

	// Each thread moves 30 elements from global
	// Three rows in original input with row padding
	__shared__ T tile[480][16+1];

#pragma unroll 10
	for(int i = 0; i < 30; i++){
		// global offsets
		unsigned int linear = input_tile_start_pos + thread_linear + (i << 8);
        	unsigned int channel_linear = linear >> 4;
        	unsigned int channel_n = linear & 15U;
		// row/col in img
		int row = channel_linear / 160;          // 0:161
		int col = channel_linear - (row * 160);  // 0:160
		// int row_padding = row - 1;
		// int col_padding = col - 1;

		// tile offsets
		// TODO: fewer the variables
		unsigned int tile_linear = linear - input_tile_start_pos;
		unsigned int row_tile = (tile_linear >> 4) / 160;
		unsigned int col_tile = (tile_linear >> 4) % 160;

		tile[row_tile * 160U + col_tile][channel_n] = !((row < 1) || (row >= 161))
					          ?input[linear - (10U << 8)]
				       	          :static_cast<T>(0);

		// tile[threadIdx.y * 60 + (threadIdx.x >> 4)][threadIdx.x & 15U] = 1?weight[weight_tile_start_pos + (i << 5) + ((threadIdx.y * 60) << 4) + threadIdx.x]:static_cast<T>(0);
	}

	__syncthreads();
	
	// Each thread moves 90 elements to global
	// TODO: Better to fuse

	// not economic
	// T val[3][12];
	// for(int i = -1, )

	for(int i = 0; i < 90; i++){
		// global offsets
		unsigned int linear = output_tile_start_pos + thread_linear + (i << 8);
		// row/col in output
		// unsigned int row = linear / 144;       // 0:25600
		// unsigned int col = linear - row * 144; // 0:144
		
		// tile offsets in output
		int linear_output_tile = linear - output_tile_start_pos;
		int row_tile = linear_output_tile / 144; // 0:160
		int col_tile = linear_output_tile - row_tile * 144; // 0:144
		int channel_n = (col_tile / 9);   // 0:16
		int channel_linear = col_tile - channel_n * 9;  // 0:9
		D[linear] = !(
			          (row_tile==0 && channel_linear%3==0) 
			       || (row_tile==159 && channel_linear%3==2)
			     )
			    ?tile[row_tile + channel_linear / 3 * 160 + (channel_linear % 3) - 1][channel_n]
			    :static_cast<T>(0);
	}
	
}


template<typename T>
void CPU_Conv_MxNxK_SiLU(const T * input, const T * weight, const T * bias, \
		T * D, int M, int N, int K)
{
	for (int i = 0; i < M * N; ++i) {
        	D[i] = 0;
        }
	for(int m = 0; m < M; m++){
		for(int n = 0; n < N; n++){
			T sum{0};
			for(int k = 0; k < K; k++){
				sum += input[m * K + k] \
						 * weight[k * N + n];
			}
			sum += bias[n];
			D[m * N + n] = silu<float>(sum);
		}

	}

	return;
}


template<typename T>
__global__ void Conv_25600x16x32_SiLU(const T * input, const T * weight, \
                const T * bias, T * D)
{

	// Input is a matrix transposed from img
	//
	// Matrix size: 25600 * 32
	//
	// Multiplying of a single img is divided into 8
	// asyn processes on separate streams indexed by
	// offset

	// GEMM
	// -----------------------------------------------
	// M = 3200, N = 16, K = 32
	// Tiling size = M: 16, N: 16, K: 16
	// TODO: try tiling size M: 16, N: 16, K: 8
	//                    or M: 32, N: 16, K: 8
	//                    or M: 32, N: 16, K: 16
	//                    or M: 64, N: 16, K: 16
	// Tiling dim: tild_M = 100, tile_N = 2, tile_K = 1
	

	// default block size 16 * 16
	// each block tile compute a 16 * 16 tile

	// unsigned int start_pos = (offset * 3200U) << 5;

	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	// block tile size 16 * 16
	// block size 16 * 16
	unsigned int block_linear = gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int thread_linear = blockDim.x * threadIdx.y + threadIdx.x;

	// 8 warps per block
	// unsigned int warp_id = (thread_linear >> 5);
	// unsigned int lane_id = thread_linear & 31U;
	
	
	// Totally 512 * 2 elements to move for each thread in 
	// block
	//
	// Two elements to each shared mem
	//
	// Do padding on the column to avoid shared mem access
	// conflict


	// Get everything required from matrix in register
	// Two accesses to global mem : Global -> Shared
	// 

	// TODO: why padding make it worse? the same for weight
	__shared__ T tiled_input[16][32];
	// __shared__ T tiled_input_transposed[32][17]
	
	// An offset of thread_linear or (* + 256) on tiled input
	tiled_input[thread_linear >> 5][thread_linear & 31U] = input[(blockIdx.y << 9)
                                                    + thread_linear];
        tiled_input[(thread_linear >> 5) + 8][thread_linear & 31U]
						            = input[(blockIdx.y << 9)
                                                    + thread_linear + 256U];

	/*
	tiled_input_transposed[threadIdx.y][threadIdx.x] = input[start_pos 
						    + block_linear * 256
						    + thread_linear];
	tiled_input_transposed[threadIdx.y][threadIdx.x + 16] = input[start_pos 
						    + block_linear * 256
						    + thread_linear + 256];
	*/
	
	
	__shared__ T tiled_weight[32][16];
	// Coalescing access
	tiled_weight[threadIdx.y][threadIdx.x] = weight[thread_linear];
	tiled_weight[threadIdx.y + 16U][threadIdx.x] = weight[thread_linear + 256U];

	__syncthreads();

	// 
	// First trying a warp tiling of size 4 * 2
	
	// No warp tiling for now
	// unsigned int warp_tile_id_x = warp_id >> 1;
	// unsigned int warp_tile_id_y = warp_id & 1;
	
	T sum{static_cast<T>(0)};

#pragma unroll
	for(int i = 0; i < 32; i++){
		sum += tiled_weight[i][threadIdx.x] * tiled_input[threadIdx.y][i];
	}
	
	sum += bias[col];
	sum = silu<T>(sum);

	// input_size : output_size = 2:1, so it the offset
	D[(row << 4)+ col] = sum;

	return;
}



template<typename T>
__global__ void Conv_25600x32x32_SiLU(const T * input, const T * weight, \
                const T * bias, T * D)
{
	// Input is a matrix transposed from img
        //
        // Matrix size: 25600 * 32
        //
        // Multiplying of a single img is divided into 8
        // asyn processes on separate streams indexed by
        // offset

	// Typically 32 * 8 blocksize
	// Each block compute 32 * 32 elements in output
		
	// unsigned int start_pos = (offset * 3200U) << 5;

        // unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
        // unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

        // block tile size 32 * 32

	// grid size dim3(1, 100) 
        // block size dim3(32,  8) PS: dim3(x_dim, y_dim, z_dim)
        unsigned int block_linear = gridDim.x * blockIdx.y + blockIdx.x;
        unsigned int thread_linear = blockDim.x * threadIdx.y + threadIdx.x;

	// Thread tile 2 * 2
	// Thread tile dim 16 * 16
        unsigned int tile_row_start = ((thread_linear >> 4) << 1);
        unsigned int tile_col_start = ((thread_linear & 15U) << 1);
	
	// 8 warps per block
        unsigned int warp_id = (thread_linear >> 5);
        unsigned int lane_id = thread_linear & 31U;

	// To compute a 32 * 32 block tile, it requires 32 * 32 from A
	// and 32 * 32 elements from B, each thread move 4 elements from
	// each
	
	// TODO: padding?	
	__shared__ T A_tile[32][32];
	__shared__ T B_tile[32][32];

	A_tile[warp_id][lane_id] = input[(blockIdx.y << 10)
						    + thread_linear];
	A_tile[warp_id + 8][lane_id] = input[(blockIdx.y << 10)
						        + thread_linear + 256];
	A_tile[warp_id + 16][lane_id] = input[(blockIdx.y << 10)
                                                        + thread_linear + 512];
	A_tile[warp_id + 24][lane_id] = input[+ (blockIdx.y << 10)
                                                        + thread_linear + 768];

	B_tile[warp_id][lane_id] = weight[thread_linear];
	B_tile[warp_id + 8][lane_id] = weight[thread_linear + 256];
	B_tile[warp_id + 16][lane_id] = weight[thread_linear + 512];
	B_tile[warp_id + 24][lane_id] = weight[thread_linear + 768];

	__syncthreads();

	T A_val[2][32];
	T B_val[32][2];

#pragma unroll
	for(int i = 0; i < 32; i++){
		// a bit away from broadcasting (TODO: 2 * 1 thread tiling?)
		A_val[0][i] = A_tile[tile_row_start][i];
		A_val[1][i] = A_tile[tile_row_start + 1][i];

		// stride = 2
		B_val[i][0] = B_tile[i][tile_col_start];
		B_val[i][1] = B_tile[i][tile_col_start + 1];
	}	

	/*T sum[2][2] = {static_cast<T>(0), static_cast<T>(0), 
		       static_cast<T>(0), static_cast<T>(0)};*/

	T sum00{};
	T sum01{};
	T sum10{};
	T sum11{};
	
#pragma unroll
	for(int i = 0; i < 32; i++){
		sum00 += A_val[0][i] * B_val[i][0];
		sum01 += A_val[0][i] * B_val[i][1];
		sum10 += A_val[1][i] * B_val[i][0];
		sum11 += A_val[1][i] * B_val[i][1];
	}

	sum00 += bias[tile_col_start];
	sum10 += bias[tile_col_start];
	sum01 += bias[tile_col_start + 1];
	sum11 += bias[tile_col_start + 1];

	sum00 = silu<T>(sum00);
	sum01 = silu<T>(sum01);
	sum10 = silu<T>(sum10);
	sum11 = silu<T>(sum11);

	// output and input are with the same size
	D[(blockIdx.y << 10) + (tile_row_start << 5)
					      + tile_col_start] = sum00;

	D[(blockIdx.y << 10) + (tile_row_start << 5)
					      + tile_col_start + 1] = sum01;
	D[(blockIdx.y << 10) + ((tile_row_start + 1) << 5)
					      + tile_col_start] = sum10;
	D[(blockIdx.y << 10) + ((tile_row_start + 1) << 5)
					      + tile_col_start + 1] = sum11;
}

template<typename T>
__global__ void concat_25600x16_25600x16_25600x32(const T * input1, const T * input2, T * D){
	// grid(800, 1) block(32, 1) for 16


	__shared__ T cache[2][48];

	// region upper bound for block
	unsigned int start_row = (blockIdx.x << 5);


#pragma unroll
	for(int i = 0; i < 16; i++){
		cache[0][threadIdx.x] = input1[(start_row << 4) + (i << 5) + threadIdx.x];
		cache[1][threadIdx.x] = input2[(start_row << 4) + (i << 5) + threadIdx.x];

		__syncthreads();
		D[(start_row << 5) + (i << 6) + threadIdx.x] = cache[threadIdx.x >> 4][(threadIdx.x) & 15U];
		D[(start_row << 5) + (i << 6) + threadIdx.x + 32] = cache[threadIdx.x >> 4][(threadIdx.x & 15U) + 16];
	
		__syncthreads();
	}
	
}


template<typename T>
void C3(const T * img, T * input, const T * weights, const T * biases, T * D, T * buffer1, T * buffer2, T * buffer3,  T * reshaped_mat, T * reshaped_weight, T * out_img){
	cudaStream_t s1, s2, s3, s4, s5, s6;
	cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&s4, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&s5, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&s6, cudaStreamNonBlocking);

	cudaEvent_t e1, e2, e3, e4, e5, e6;
	cudaEventCreateWithFlags(&e1, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&e2, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&e3, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&e4, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&e5, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&e6, cudaEventDisableTiming);

	dim3 blocksize_im2col(32, 32);	
	dim3 gridsize_im2col(800, 1);	
	im2col_32x160x160_25600x32_transpose<float>
					    <<<gridsize_im2col, blocksize_im2col, 0, s1>>>
					    (img, input);

	
	CHECK_LAST_CUDA_ERROR();
	
	dim3 blocksize_wreshape(16, 16);	
	dim3 gridsize_wreshape(9, 1);	
	Convm1_weight_reshape_16x16x3x3_144x16<float>
					    <<<gridsize_wreshape, blocksize_wreshape, 0, s2>>>
					    ((float *)(weights + CONV_WEIGHT_m1_OFFSET),
					     reshaped_weight);


	CHECK_LAST_CUDA_ERROR();
	
	CHECK_CUDA_ERROR(cudaEventRecord(e1, s1));
	// CHECK_CUDA_ERROR(cudaEventSynchronize(e1));
	CHECK_CUDA_ERROR(cudaStreamWaitEvent(s3, e1, 0));
	CHECK_CUDA_ERROR(cudaStreamWaitEvent(s4, e1, 0));


	dim3 blocksize_fused(16, 16);
        dim3 gridsize_fused(1, 800);
	fused_25600x16x32_25600x16x16_SiLU<float>
						 <<<gridsize_fused, blocksize_fused, 0, s3>>>
						 ( input,
						  (float *)(weights + CONV_WEIGHT_1_OFFSET),
						  (float *)(biases + CONV_BIAS_1_OFFSET),
						  (float *)(weights + CONV_WEIGHT_m0_OFFSET),
						  (float *)(biases + CONV_BIAS_m0_OFFSET),
						  buffer3,
						  buffer1
						  );


	CHECK_LAST_CUDA_ERROR();
	
	dim3 blocksize_conv2(16, 16);
        dim3 gridsize_conv2(1, 1600);
        Conv_25600x16x32_SiLU<float><<<gridsize_conv2, blocksize_conv2, 0, s4>>>
                                        ( input,
                                         (float *)(weights + CONV_WEIGHT_2_OFFSET),
                                         (float *)(biases + CONV_BIAS_2_OFFSET),
					 buffer2);


	CHECK_LAST_CUDA_ERROR();
	
	CHECK_CUDA_ERROR(cudaEventRecord(e4, s4));
	// CHECK_CUDA_ERROR(cudaEventSynchronize(e4));
	
	dim3 blocksize_ireshape(32, 8);
        dim3 gridsize_ireshape(160, 1);
	Convm1_input_reshape_25600x16_25600x144<float>
					       <<<gridsize_ireshape, blocksize_ireshape, 0, s3>>>
					       (buffer1,
						reshaped_mat);

	CHECK_LAST_CUDA_ERROR();
	
	CHECK_CUDA_ERROR(cudaEventRecord(e2, s2));
	// CHECK_CUDA_ERROR(cudaEventSynchronize(e2));
	CHECK_CUDA_ERROR(cudaEventRecord(e3, s3));
	// CHECK_CUDA_ERROR(cudaEventSynchronize(e3));
	CHECK_CUDA_ERROR(cudaStreamWaitEvent(s5, e2, 0));
	CHECK_CUDA_ERROR(cudaStreamWaitEvent(s5, e3, 0));

	dim3 blocksize_convm1(16, 16);
        dim3 gridsize_convm1(800, 1);
	Convm1_25600x144x16_SiLU_adding<float><<<gridsize_convm1, blocksize_convm1, 0, s5>>>
				       (reshaped_mat,
					reshaped_weight,
					(float*)(biases + CONV_BIAS_m1_OFFSET),
					buffer3,
					buffer1);
	
	CHECK_LAST_CUDA_ERROR();
	
	CHECK_CUDA_ERROR(cudaEventRecord(e5, s5));
	// CHECK_CUDA_ERROR(cudaEventSynchronize(e5));
	CHECK_CUDA_ERROR(cudaStreamWaitEvent(s6, e4, 0));
	CHECK_CUDA_ERROR(cudaStreamWaitEvent(s6, e5, 0));


	dim3 blocksize_cct(32, 1);
        dim3 gridsize_cct(800, 1);
	concat_25600x16_25600x16_25600x32<float>
					 <<<gridsize_cct, blocksize_cct, 0, s6>>>
					 (buffer1,
					  buffer2,
					  buffer3
					  );

	CHECK_LAST_CUDA_ERROR();


	dim3 blocksize_conv3(32, 8);
        dim3 gridsize_conv3(1, 800);
	Conv_25600x32x32_SiLU<float><<<gridsize_conv3, blocksize_conv3, 0, s6>>>
				    (buffer3,
				     (float*) (weights + CONV_WEIGHT_3_OFFSET),
				     (float*) (biases + CONV_BIAS_3_OFFSET),
				     D);

	CHECK_LAST_CUDA_ERROR();

	dim3 blocksize_col2im(32, 32);
        dim3 gridsize_col2im(800, 1);
	col2im_25600x32_32x160x160_transpose<float>
					    <<<gridsize_col2im, blocksize_col2im, 0, s6>>>
					    (D,
					     out_img);

	CHECK_LAST_CUDA_ERROR();
	
	CHECK_CUDA_ERROR(cudaEventRecord(e6, s6));
	CHECK_CUDA_ERROR(cudaEventSynchronize(e6));
	
	CHECK_CUDA_ERROR(cudaStreamDestroy(s1));
	CHECK_CUDA_ERROR(cudaStreamDestroy(s2));
	CHECK_CUDA_ERROR(cudaStreamDestroy(s3));
	CHECK_CUDA_ERROR(cudaStreamDestroy(s4));
	CHECK_CUDA_ERROR(cudaStreamDestroy(s5));
	CHECK_CUDA_ERROR(cudaStreamDestroy(s6));
	
	CHECK_CUDA_ERROR(cudaEventDestroy(e1));
	CHECK_CUDA_ERROR(cudaEventDestroy(e2));
	CHECK_CUDA_ERROR(cudaEventDestroy(e3));
	CHECK_CUDA_ERROR(cudaEventDestroy(e4));
	CHECK_CUDA_ERROR(cudaEventDestroy(e5));
	CHECK_CUDA_ERROR(cudaEventDestroy(e6));


}


// ptr in out
void transpose_weight(float* p, int K, int N) {
        if (K <= 0 || N <= 0) {
                return;
        }

        const size_t rows = static_cast<size_t>(K);
        const size_t cols = static_cast<size_t>(N);
        std::vector<float> buffer(rows * cols);

        for (size_t r = 0; r < rows; ++r) {
                for (size_t c = 0; c < cols; ++c) {
                        buffer[c * rows + r] = p[r * cols + c];
                }
        }

        std::memcpy(p, buffer.data(), buffer.size() * sizeof(float));
}


int main(int arg, char ** args){
	bool test_flags[5];
	bool C3_test = args[1][0]=='1';

	if(arg>=3){
		for(int i = 0; i < 5; i++){
			test_flags[i] = (args[2][i]=='1');
		}
	
	}
	if(C3_test){
	float * img, * weights, * biases, * buffer1, *buffer2, * buffer3, * reshaped_mat, * reshaped_bias, * out_img;
	CHECK_CUDA_ERROR(cudaHostAlloc((void**)&img, INPUT_SIZE * sizeof(float), cudaHostAllocDefault));
	// CHECK_CUDA_ERROR(cudaHostAlloc((void**)&input, INPUT_SIZE * sizeof(float), cudaHostAllocDefault));
	CHECK_CUDA_ERROR(cudaHostAlloc((void**)&weights, CONV_WEIGHT_SIZE * sizeof(float), cudaHostAllocDefault));
	CHECK_CUDA_ERROR(cudaHostAlloc((void**)&biases, CONV_BIAS_SIZE * sizeof(float), cudaHostAllocDefault));
	// CHECK_CUDA_ERROR(cudaHostAlloc((void**)&output, OUTPUT_SIZE * sizeof(float), cudaHostAllocDefault));

	// Use pinned mem instead
	// float * input = (float *)malloc(INPUT_SIZE * sizeof(float));
	// float * weights = (float *)malloc(CONV_WEIGHT_SIZE * sizeof(float));
	// float * biases = (float *)malloc(CONV_BIAS_SIZE * sizeof(float));
	// float * output = (float *)malloc(OUTPUT_SIZE * sizeof(float));

	// Temp and Output utilization;
	// Input --Conv1--> [Temp + 0u] --Convm0,Convm1--> [Temp + OUTPUT_SIZE >> 1]
	//                            \-------------------------------\ --Add--
        //                                             [Output + 0u] <--Move--/
	// Input --Conv2--> [Output + OUTPUT_SIZE >> 1]

	
	// Use pinned mem instead
	// f:qloat * buffer = (float *)malloc(OUTPUT_SIZE * sizeof(float)); 
	// CHECK_CUDA_ERROR(cudaHostAlloc((void**)&buffer1, OUTPUT_SIZE * sizeof(float), cudaHostAllocDefault));
	// CHECK_CUDA_ERROR(cudaHostAlloc((void**)&buffer2, OUTPUT_SIZE * sizeof(float), cudaHostAllocDefault));
	// CHECK_CUDA_ERROR(cudaHostAlloc((void**)&buffer3, OUTPUT_SIZE * sizeof(float) << 1, cudaHostAllocDefault));
	// CHECK_CUDA_ERROR(cudaHostAlloc((void**)&reshaped_mat, 9U * OUTPUT_SIZE * sizeof(float) >> 1, cudaHostAllocDefault));
	// CHECK_CUDA_ERROR(cudaHostAlloc((void**)&reshaped_weight, (9U << 8) * sizeof(float), cudaHostAllocDefault));
	CHECK_CUDA_ERROR(cudaHostAlloc((void**)&out_img, OUTPUT_SIZE * sizeof(float), cudaHostAllocDefault));
	

	load_npy_into<float>("parameters/cv1.conv.bias", (float *)(biases + CONV_BIAS_1_OFFSET), CONV_BIAS_1_SIZE);
	load_npy_into<float>("parameters/cv2.conv.bias", (float *)(biases + CONV_BIAS_2_OFFSET), CONV_BIAS_2_SIZE);
	load_npy_into<float>("parameters/cv3.conv.bias", (float *)(biases + CONV_BIAS_3_OFFSET), CONV_BIAS_3_SIZE);
	load_npy_into<float>("parameters/m.0.cv1.conv.bias", (float *)(biases + CONV_BIAS_m0_OFFSET), CONV_BIAS_m0_SIZE);
	load_npy_into<float>("parameters/m.0.cv2.conv.bias", (float *)(biases + CONV_BIAS_m1_OFFSET), CONV_BIAS_m1_SIZE);

	load_npy_into<float>("parameters/cv1.conv.weight", (float *)(weights + CONV_WEIGHT_1_OFFSET), CONV_WEIGHT_1_SIZE);
	transpose_weight((float *)(weights + CONV_WEIGHT_1_OFFSET), 32, 16);
	load_npy_into<float>("parameters/cv2.conv.weight", (float *)(weights + CONV_WEIGHT_2_OFFSET), CONV_WEIGHT_2_SIZE);
	transpose_weight((float *)(weights + CONV_WEIGHT_2_OFFSET), 32, 16);
	load_npy_into<float>("parameters/cv3.conv.weight", (float *)(weights + CONV_WEIGHT_3_OFFSET), CONV_WEIGHT_3_SIZE);
	transpose_weight((float *)(weights + CONV_WEIGHT_3_OFFSET), 32, 32);
	load_npy_into<float>("parameters/m.0.cv1.conv.weight", (float *)(weights + CONV_WEIGHT_m0_OFFSET), CONV_WEIGHT_m0_SIZE);
	transpose_weight((float *)(weights + CONV_WEIGHT_m0_OFFSET), 16, 16);
	load_npy_into<float>("parameters/m.0.cv2.conv.weight", (float *)(weights + CONV_WEIGHT_m1_OFFSET), CONV_WEIGHT_m1_SIZE);

	/* for(int i = 0; i < CONV_WEIGHT_m1_SIZE; i++){
		std::cout << *(float *)(weights + CONV_WEIGHT_m1_OFFSET + i) << " ";
	}
	std::cout << std::endl; */

	// loading input
	

	load_input_into<float>("data/inputs/input_0.txt", img, INPUT_SIZE);
	/* for(int i = 0; i < 16; i++){
		for(int j = 0; j < 16; j++){
			std::cout << input[i * 25600 + j] << " ";
		}
		std::cout << std::endl;
	} */
	

	float * d_img, * d_input, * d_weights, * d_biases, * d_output, * d_buffer1, * d_buffer2, * d_buffer3, * d_temp_input_1, * d_temp_input_2, * d_out_img, * d_reshaped_mat, * d_reshaped_weight; 

	// Start timing


	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_img, INPUT_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(float)));




	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_temp_input_1, INPUT_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_temp_input_2, INPUT_SIZE * sizeof(float)));
	


	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights, CONV_WEIGHT_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_biases, CONV_BIAS_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_buffer1, OUTPUT_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_buffer2, OUTPUT_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_buffer3, OUTPUT_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out_img, OUTPUT_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_reshaped_mat, 9U * OUTPUT_SIZE * sizeof(float) >> 1));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_reshaped_weight, (9U << 8) * sizeof(float)));


	// Or start timing here?
	CHECK_CUDA_ERROR(cudaMemcpy(d_img, img, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_weights, weights, CONV_WEIGHT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_biases, biases, CONV_BIAS_SIZE * sizeof(float), cudaMemcpyHostToDevice));

	/* dim3 blocksize(32, 32);
        dim3 gridsize(800, 1);
        im2col_32x160x160_25600x32_transpose<float><<<gridsize, blocksize>>>(d_input, d_temp_input_1);
	
	CHECK_LAST_CUDA_ERROR();
	
        col2im_25600x32_32x160x160_transpose<float><<<gridsize, blocksize>>>(d_temp_input_1, d_temp_input_2);

	CHECK_LAST_CUDA_ERROR();*/

	float * temp_input_1 = (float*)malloc(100U * sizeof(float) << 13);
	float * temp_input_2 = (float*)malloc(100U * sizeof(float) << 13);


	CHECK_CUDA_ERROR(cudaMemcpy(temp_input_1, d_temp_input_1, 100U * sizeof(float) << 13, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(temp_input_2, d_temp_input_2, 100U * sizeof(float) << 13, cudaMemcpyDeviceToHost));
	

	/* int transpose_miss_1 = 0;
	int transpose_miss_2 = 0;
	for(int i = 0; i < 32; i++){
		for(int j = 0; j < 25600; j++){
			if(abs(temp_input_1[j * 32 + i] - input[i * 25600 + j]) > 0.0001f){
				transpose_miss_1++;
			}
		}
	}

	for(int i = 0; i < 25600 * 32; i++){
		if(abs(temp_input_2[i] - input[i]) > 0.0001f){
			transpose_miss_2++;
		}
	}
	std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
	std::cout << transpose_miss_1 << std::endl;
	std::cout << transpose_miss_2 << std::endl;*/

	CHECK_CUDA_ERROR(cudaFree(d_temp_input_1));
	CHECK_CUDA_ERROR(cudaFree(d_temp_input_2));
	free(temp_input_1);
	free(temp_input_2);

	// TODO: finish C3 Test
	cudaEvent_t C3_start, C3_stop;
	float C3_runtime;
	CHECK_CUDA_ERROR(cudaEventCreate(&C3_start));
	CHECK_CUDA_ERROR(cudaEventCreate(&C3_stop));

	CHECK_CUDA_ERROR(cudaEventRecord(C3_start, 0));
        
	C3<float>(d_img, d_input, d_weights, d_biases, d_output, d_buffer1, d_buffer2, d_buffer3, d_reshaped_mat, d_reshaped_weight, d_out_img);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	CHECK_CUDA_ERROR(cudaEventRecord(C3_stop, 0));
	CHECK_CUDA_ERROR(cudaEventSynchronize(C3_stop));
	CHECK_CUDA_ERROR(cudaEventElapsedTime(&C3_runtime, C3_start, C3_stop));
	
	std::cout << "C3 runtime: " << C3_runtime << " ms" << std::endl;

	CHECK_LAST_CUDA_ERROR();

        // CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	CHECK_CUDA_ERROR(cudaMemcpy(out_img, d_out_img, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

	for(int i = 0; i < 32; i++){
		std::cout << "Channel " << i << " first 5: ";
		for(int j = 0; j < 5; j ++){
			std::cout << out_img[i * 25600 + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;


        CHECK_CUDA_ERROR(cudaFree(d_input));
        CHECK_CUDA_ERROR(cudaFree(d_weights));
        CHECK_CUDA_ERROR(cudaFree(d_biases));
        CHECK_CUDA_ERROR(cudaFree(d_output));
        CHECK_CUDA_ERROR(cudaFree(d_buffer1));
        CHECK_CUDA_ERROR(cudaFree(d_buffer2));
        CHECK_CUDA_ERROR(cudaFree(d_buffer3));
        CHECK_CUDA_ERROR(cudaFree(d_out_img));

        CHECK_CUDA_ERROR(cudaFreeHost(img));
        // CHECK_CUDA_ERROR(cudaFreeHost(input));
        CHECK_CUDA_ERROR(cudaFreeHost(weights));
        CHECK_CUDA_ERROR(cudaFreeHost(biases));
        // CHECK_CUDA_ERROR(cudaFreeHost(output));
        CHECK_CUDA_ERROR(cudaFreeHost(out_img));
        // CHECK_CUDA_ERROR(cudaFreeHost(buffer1));
        // CHECK_CUDA_ERROR(cudaFreeHost(buffer2));
        // CHECK_CUDA_ERROR(cudaFreeHost(buffer3));

        // Use pinned memory instead
        // free(input); free(weights); free(biases); free(output); free(buffer);

        CHECK_CUDA_ERROR(cudaDeviceReset());
        } 

	if(arg > 2){
        
	

	// Unit tests
	std::cout << "-----------------------------------------------" 
		<< std::endl
		<< "Unit tests begin" << std::endl
                << "-----------------------------------------------"
                << std::endl
		<< std::endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
		

		
	float * t_input_1, * t_weight_1, * t_bias_1, * t_output_1, * t_gt_1;
	float * d_t_input_1, * d_t_weight_1, * d_t_bias_1, * d_t_output_1;
		
	float * t_input_2, * t_weight_2, * t_bias_2, * t_output_2, * t_gt_2;
	float * d_t_input_2, * d_t_weight_2, * d_t_bias_2, * d_t_output_2;

	float runtime;

	if(test_flags[0]){	
		// Test for Conv 3200 * 32 * 32 with Gelu activation
		 
		std::cout << "-----------------------------------------------" << std::endl;
		std::cout << "Unit Test 1 on Conv3 begins." << std::endl;
		std::cout << "-----------------------------------------------"
			  << std::endl;

		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_input_1, 100U * sizeof(float) << 13, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_weight_1, sizeof(float) << 10, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_bias_1, sizeof(float) << 5, cudaHostAllocDefault));

		

		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_output_1, 100U * sizeof(float) << 13, cudaHostAllocDefault));
		t_gt_1 = (float *)malloc(100U * sizeof(float) << 13);
		memset((void*) t_gt_1, 0, 100U * sizeof(float) << 13);

		for(int i = 0; i < 819200; i++){
			t_input_1[i] = static_cast<float>(rand()) \
				       / static_cast<float>(RAND_MAX) - 0.5f;
		}

		for(int i = 0; i < 1024; i++){
			t_weight_1[i] = static_cast<float>(rand()) \
                                       / static_cast<float>(RAND_MAX) - 0.5f;
		}

		for(int i = 0; i < 32; i++){
                        t_bias_1[i] = static_cast<float>(rand()) \
                                       / static_cast<float>(RAND_MAX);
                }

		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_input_1, 100U * sizeof(float) << 13));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_weight_1, sizeof(float) << 10));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_bias_1, sizeof(float) << 5));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_output_1, 100U * sizeof(float) << 13));
			
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_input_1, t_input_1, 100U * \ 
					sizeof(float) << 13, cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_weight_1, t_weight_1, \
					sizeof(float) << 10,  cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_bias_1, t_bias_1, \ 
                                        sizeof(float) << 5, cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemset((void*)d_t_output_1, 0, 100U * sizeof(float) << 13));

		dim3 block_size_1(32, 8);
		dim3 grid_size_1(1, 800);

		Conv_25600x32x32_SiLU<float>
                        <<<grid_size_1, block_size_1>>>
                                (d_t_input_1, \
                                 d_t_weight_1, \
                                 d_t_bias_1, \
                                 d_t_output_1);
		Conv_25600x32x32_SiLU<float>
                        <<<grid_size_1, block_size_1>>>
                                (d_t_input_1, \
                                 d_t_weight_1, \
                                 d_t_bias_1, \
                                 d_t_output_1);
		// Timed run
		cudaEventRecord(start);
		Conv_25600x32x32_SiLU<float>
			<<<grid_size_1, block_size_1>>>
				(d_t_input_1, \
				 d_t_weight_1, \
				 d_t_bias_1, \
			         d_t_output_1);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));	
		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&runtime, start, stop));

		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaMemcpy(t_output_1, d_t_output_1, 100U * \
					sizeof(float) << 13, cudaMemcpyDeviceToHost));

		CPU_Conv_MxNxK_SiLU(t_input_1, t_weight_1, t_bias_1, t_gt_1, 25600, 32, 32);	

		bool flag1 = true;
		int false_num = 0;
		for(int i = 0; i < 25600; i++){
			for(int j = 0; j < 32; j++){
				/* if(i >= 0 && i <= 3 && j >= 0 && j <= 3){
					std::cout << t_gt_1[i * 32 + j] << " " << t_output_1[i * 32 + j] << std::endl;
				} */
				if(abs(t_gt_1[i * 32 + j] - t_output_1[i * 32 + j]) > 0.0001f){
					flag1 = false;
					false_num++;
				}
			}
		}
		std::cout << "Test 1 passed?: " << flag1 << std::endl;
		std::cout << "Mistake on "<< false_num << " elements out of 819200" << std::endl;
		std::cout << "Test 1 runtime: " << runtime * 1000 << " us" << std::endl;
		
	       	CHECK_CUDA_ERROR(cudaFreeHost(t_input_1));	
	       	CHECK_CUDA_ERROR(cudaFreeHost(t_weight_1));	
	       	CHECK_CUDA_ERROR(cudaFreeHost(t_bias_1));	
	       	CHECK_CUDA_ERROR(cudaFreeHost(t_output_1));

	       	CHECK_CUDA_ERROR(cudaFree(d_t_input_1));	
	       	CHECK_CUDA_ERROR(cudaFree(d_t_weight_1));	
	       	CHECK_CUDA_ERROR(cudaFree(d_t_bias_1));	
	       	CHECK_CUDA_ERROR(cudaFree(d_t_output_1));

		free(t_gt_1);



		std::cout << "Unit Test 1 on Conv3 done." << std::endl
                        << std::endl;
		std::cout << std::endl;
		
	}

	if(test_flags[1]){
		// Test for Conv 25600 * 16 * 32 with Gelu activation
		std::cout << "-----------------------------------------------" << std::endl;
		std::cout << "Unit Test 2 on Conv1 begins." << std::endl;
		std::cout << "-----------------------------------------------"
			  << std::endl;


		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_input_2, 100U * sizeof(float) << 13, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_weight_2, sizeof(float) << 9, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_bias_2, sizeof(float) << 4, cudaHostAllocDefault));

		

		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_output_2, 100U * sizeof(float) << 12, cudaHostAllocDefault));
		t_gt_2 = (float *)malloc(100U * sizeof(float) << 12);
		memset((void*) t_gt_2, 0, 100U * sizeof(float) << 12);

		for(int i = 0; i < 819200; i++){
			t_input_2[i] = static_cast<float>(rand()) \
				       / static_cast<float>(RAND_MAX) - 0.5f;
		}

		for(int i = 0; i < 512; i++){
			t_weight_2[i] = static_cast<float>(rand()) \
                                       / static_cast<float>(RAND_MAX) - 0.5f;
		}

		for(int i = 0; i < 16; i++){
                        t_bias_2[i] = static_cast<float>(rand()) \
                                       / static_cast<float>(RAND_MAX);
                }

		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_input_2, 100U * sizeof(float) << 13));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_weight_2, sizeof(float) << 9));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_bias_2, sizeof(float) << 4));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_output_2, 100U * sizeof(float) << 12));
			
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_input_2, t_input_2, 100U * \ 
					sizeof(float) << 13, cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_weight_2, t_weight_2, \
					sizeof(float) << 9,  cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_bias_2, t_bias_2, \ 
                                        sizeof(float) << 4, cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemset((void*)d_t_output_2, 0, 100U * sizeof(float) << 12));

		dim3 block_size_2(16, 16);
		dim3 grid_size_2(1, 1600);
		Conv_25600x16x32_SiLU<float>
			<<<grid_size_2, block_size_2>>>
				(d_t_input_2, \
				 d_t_weight_2, \
				 d_t_bias_2, \
			         d_t_output_2);
		Conv_25600x16x32_SiLU<float>
                        <<<grid_size_2, block_size_2>>>
                                (d_t_input_2, \
                                 d_t_weight_2, \
                                 d_t_bias_2, \
                                 d_t_output_2);
		
		// Timed run
		cudaEventRecord(start);
		Conv_25600x16x32_SiLU<float>
                        <<<grid_size_2, block_size_2>>>
                                (d_t_input_2, \
                                 d_t_weight_2, \
                                 d_t_bias_2, \
                                 d_t_output_2);
		cudaEventRecord(stop);
    		cudaEventSynchronize(stop);
    		cudaEventElapsedTime(&runtime, start, stop);
		

		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaMemcpy(t_output_2, d_t_output_2, 100U * \
					sizeof(float) << 12, cudaMemcpyDeviceToHost));

		CPU_Conv_MxNxK_SiLU(t_input_2, t_weight_2, t_bias_2, t_gt_2, 25600, 16, 32);	

		bool flag2 = true;
		int false_num = 0;
		for(int i = 0; i < 25600; i++){
			for(int j = 0; j < 16; j++){
				/* if(i >= 0 && i <= 3 && j >= 0 && j <= 3){
					std::cout << t_gt_2[i * 16 + j] << " " << t_output_2[i * 16 + j] << std::endl;
				} */
				if(abs(t_gt_2[i * 16 + j] - t_output_2[i * 16 + j]) > 0.0001f){
					flag2 = false;
					false_num++;
				}
			}
		}
		std::cout << "Test 2 passed?: " << flag2 << std::endl;
		std::cout << "Mistake on "<< false_num << " elements out of 409600" << std::endl;
		std::cout << "Test 2 runtime: " << runtime * 1000 << " us" << std::endl;
		
	       	CHECK_CUDA_ERROR(cudaFreeHost(t_input_2));	
	       	CHECK_CUDA_ERROR(cudaFreeHost(t_weight_2));	
	       	CHECK_CUDA_ERROR(cudaFreeHost(t_bias_2));	
	       	CHECK_CUDA_ERROR(cudaFreeHost(t_output_2));

	       	CHECK_CUDA_ERROR(cudaFree(d_t_input_2));	
	       	CHECK_CUDA_ERROR(cudaFree(d_t_weight_2));	
	       	CHECK_CUDA_ERROR(cudaFree(d_t_bias_2));	
	       	CHECK_CUDA_ERROR(cudaFree(d_t_output_2));

		free(t_gt_2);	
		std::cout << "Unit Test 2 on Conv1 done." << std::endl
			<< std::endl;
		std::cout << std::endl;
	}


	if(test_flags[2]){

		// Test for Fused Conv 3200 * 32 * 32 with Gelu activation
		// and Conv 3200 * 16 * 16 with Gelu activation
		std::cout << "-----------------------------------------------" << std::endl;
		std::cout << "Unit Test 3 on fused layers begins." << std::endl;
                std::cout << "-----------------------------------------------"
                          << std::endl;
		
		
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_input_1, 100U * sizeof(float) << 13, cudaHostAllocDefault));
                CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_weight_1, sizeof(float) << 9, cudaHostAllocDefault));
                CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_weight_2, sizeof(float) << 8, cudaHostAllocDefault));
                CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_bias_1, sizeof(float) << 4, cudaHostAllocDefault));
                CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_bias_2, sizeof(float) << 4, cudaHostAllocDefault));



                CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_output_1, 100U * sizeof(float) << 12, cudaHostAllocDefault));
                CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_output_2, 100U * sizeof(float) << 12, cudaHostAllocDefault));
                t_gt_1 = (float *)malloc(100U * sizeof(float) << 12);
                t_gt_2 = (float *)malloc(100U * sizeof(float) << 12);
                memset((void*) t_gt_1, 0, 100U * sizeof(float) << 12);
                memset((void*) t_gt_2, 0, 100U * sizeof(float) << 12);
		
		for(int i = 0; i < 819200; i++){
                        t_input_1[i] = static_cast<float>(rand()) \
                                       / static_cast<float>(RAND_MAX) - 0.5f;
                }

                for(int i = 0; i < 512; i++){
                        t_weight_1[i] = static_cast<float>(rand()) \
                                       / static_cast<float>(RAND_MAX) - 0.5f;
                }

                for(int i = 0; i < 16; i++){
                        t_bias_1[i] = static_cast<float>(rand()) \
                                       / static_cast<float>(RAND_MAX);
                }

		for(int i = 0; i < 256; i++){
                        t_weight_2[i] = static_cast<float>(rand()) \
                                       / static_cast<float>(RAND_MAX) - 0.5f;
                }

                for(int i = 0; i < 16; i++){
                        t_bias_2[i] = static_cast<float>(rand()) \
                                       / static_cast<float>(RAND_MAX);
                }


		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_input_1, 100U * sizeof(float) << 13));
                CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_weight_1, sizeof(float) << 9));
                CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_bias_1, sizeof(float) << 4));
                CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_output_1, 100U * sizeof(float) << 12));
                CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_weight_2, sizeof(float) << 8));
                CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_bias_2, sizeof(float) << 4));
                CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_output_2, 100U * sizeof(float) << 12));

		CHECK_CUDA_ERROR(cudaMemcpy(d_t_input_1, t_input_1, 100U * \
                                        sizeof(float) << 13, cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemcpy(d_t_weight_1, t_weight_1, \
                                        sizeof(float) << 9,  cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemcpy(d_t_bias_1, t_bias_1, \
                                        sizeof(float) << 4, cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_weight_2, t_weight_2, \
                                        sizeof(float) << 8,  cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemcpy(d_t_bias_2, t_bias_2, \
                                        sizeof(float) << 4, cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemset((void*)d_t_output_1, 0, 100U * sizeof(float) << 12));
                CHECK_CUDA_ERROR(cudaMemset((void*)d_t_output_2, 0, 100U * sizeof(float) << 12));
		

		dim3 grid_size_3(1, 800);
		dim3 block_size_3(16, 16);	

		
		fused_25600x16x32_25600x16x16_SiLU<float>
                        <<<grid_size_3, block_size_3>>>
                        (d_t_input_1,
                        d_t_weight_1,
                        d_t_bias_1,
                        d_t_weight_2,
                        d_t_bias_2,
			d_t_output_1,
                        d_t_output_2
                        );

		fused_25600x16x32_25600x16x16_SiLU<float>
                        <<<grid_size_3, block_size_3>>>
                        (d_t_input_1,
                        d_t_weight_1,
                        d_t_bias_1,
                        d_t_weight_2,
                        d_t_bias_2,
			d_t_output_1,
                        d_t_output_2
                        );

		// Timed run
                cudaEventRecord(start);
		fused_25600x16x32_25600x16x16_SiLU<float>
                        <<<grid_size_3, block_size_3>>>
                        (d_t_input_1,
                        d_t_weight_1,
                        d_t_bias_1,
			d_t_weight_2,
			d_t_bias_2,
			d_t_output_1,
                        d_t_output_2
                        );	

		CHECK_LAST_CUDA_ERROR();
		
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
                CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
                CHECK_CUDA_ERROR(cudaEventElapsedTime(&runtime, start, stop));
		


		CPU_Conv_MxNxK_SiLU(t_input_1, t_weight_1, t_bias_1, t_gt_1, 25600, 16, 32);	
		CPU_Conv_MxNxK_SiLU(t_gt_1, t_weight_2, t_bias_2, t_gt_2, 25600, 16, 16);	
		
		/* for(int i = 0; i < 25600; i++){
			for(int j = 0; j < 16; j++){
				t_gt_2[(i << 4) + j] += t_input_1[(i << 4) + j];
			}
		} */

		CHECK_CUDA_ERROR(cudaMemcpy(t_output_1, d_t_output_1, 100U * \
                                        sizeof(float) << 12, cudaMemcpyDeviceToHost));

		CHECK_CUDA_ERROR(cudaMemcpy(t_output_2, d_t_output_2, 100U * \
                                        sizeof(float) << 12, cudaMemcpyDeviceToHost));
		

		bool flag3 = true;
                int false_num = 0;
                for(int i = 0; i < 25600; i++){
                        for(int j = 0; j < 16; j++){
                                /*if(i >= 0 && i <= 3 && j >= 0 && j <= 3){
                                        std::cout << t_gt_2[i * 16 + j] << " " << t_output_2[i * 16 + j] << std::endl;
                                } */
                                if(abs(t_gt_1[i * 16 + j] - t_output_1[i * 16 + j]) > 0.0001f){
                                        flag3 = false;
                                        false_num++;
                                }
				/*else{
                                        std::cout << i << " " << j << std::endl;
                                }*/
                        }
                }

		for(int i = 0; i < 25600; i++){
                        for(int j = 0; j < 16; j++){
                                /*if(i >= 0 && i <= 3 && j >= 0 && j <= 3){
                                        std::cout << t_gt_2[i * 16 + j] << " " << t_output_2[i * 16 + j] << std::endl;
                                }*/
                                if(abs(t_gt_2[i * 16 + j] - t_output_2[i * 16 + j]) > 0.0001f){
                                        flag3 = false;
                                        false_num++;
                                }
				/*else{
					std::cout << i << " " << j << std::endl;
				}*/
                        }
                }

                std::cout << "Test 3 passed?: " << flag3 << std::endl;
                std::cout << "Mistake on "<< false_num << " elements out of 409600 * 2" << std::endl;
                std::cout << "Test 3 runtime: " << runtime * 1000 << " us" << std::endl;

		CHECK_CUDA_ERROR(cudaFreeHost(t_input_1));
                CHECK_CUDA_ERROR(cudaFreeHost(t_weight_1));
                CHECK_CUDA_ERROR(cudaFreeHost(t_weight_2));
                CHECK_CUDA_ERROR(cudaFreeHost(t_bias_1));
                CHECK_CUDA_ERROR(cudaFreeHost(t_bias_2));
                CHECK_CUDA_ERROR(cudaFreeHost(t_output_1));
                CHECK_CUDA_ERROR(cudaFreeHost(t_output_2));

                CHECK_CUDA_ERROR(cudaFree(d_t_input_1));
                CHECK_CUDA_ERROR(cudaFree(d_t_weight_1));
                CHECK_CUDA_ERROR(cudaFree(d_t_weight_2));
                CHECK_CUDA_ERROR(cudaFree(d_t_bias_1));
                CHECK_CUDA_ERROR(cudaFree(d_t_bias_2));
                CHECK_CUDA_ERROR(cudaFree(d_t_output_1));
                CHECK_CUDA_ERROR(cudaFree(d_t_output_2));

                free(t_gt_1);
                free(t_gt_2);

		std::cout << "Unit Test 3 on fused layers done." << std::endl
                        << std::endl;
		std::cout << std::endl;
	}

	if(test_flags[3]){

                // Test for 3x3 kernel Conv 3200 * 16 * 16 with Gelu activation
		std::cout << "-----------------------------------------------" << std::endl;
                std::cout << "Unit Test 4 on 3x3 Conv begins." << std::endl;
                std::cout << "-----------------------------------------------"
                          << std::endl;
		
		
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_input_1, 100U * sizeof(float) << 12, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_weight_1, 9U * sizeof(float) << 8, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_bias_1, sizeof(float) << 4, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_output_1, 100U * sizeof(float) << 12, cudaHostAllocDefault));
		t_gt_1 = (float *)malloc(100U * sizeof(float) << 12);
		memset((void*) t_gt_1, 0, 100U * sizeof(float) << 12);
		
		for(int i = 0; i < 409600; i++){
                        t_input_1[i] = static_cast<float>(rand()) \
                                       / static_cast<float>(RAND_MAX) - 0.5f;
                }

		for(int i = 0; i < 2304; i++){
                        t_weight_1[i] = static_cast<float>(rand()) \
                                       / static_cast<float>(RAND_MAX) - 0.5f;
                }

		for(int i = 0; i < 16; i++){
                        t_bias_1[i] = static_cast<float>(rand()) \
                                       / static_cast<float>(RAND_MAX) - 0.5f;
                }


		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_input_1, 100U * sizeof(float) << 12));
                CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_weight_1, 9U * sizeof(float) << 8));
                CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_bias_1, sizeof(float) << 4));
                CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_output_1, 100U * sizeof(float) << 12));

		// Use async copy instead
                // CHECK_CUDA_ERROR(cudaMemcpy(d_t_input_1, t_input_1, 100U * \
                                        sizeof(float) << 12, cudaMemcpyHostToDevice));
                // CHECK_CUDA_ERROR(cudaMemcpy(d_t_weight_1, t_weight_1, 9U * \
                                        sizeof(float) << 8,  cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemcpy(d_t_bias_1, t_bias_1, \
                                        sizeof(float) << 4, cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemset((void*)d_t_output_1, 0, 100U * sizeof(float) << 12));

		float * reshaped_input, * reshaped_weight;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&reshaped_input, 900U * sizeof(float) << 12));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&reshaped_weight, 9U * sizeof(float) << 8));

		
		cudaStream_t s1, s2, s3, ctrl;
		cudaEvent_t e1, e2, e3;

		CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
		CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
		CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking));
		CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&ctrl, cudaStreamNonBlocking));

		
		CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&e1, cudaEventDisableTiming));
		CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&e2, cudaEventDisableTiming));

		CHECK_CUDA_ERROR(cudaMemcpyAsync(d_t_input_1, t_input_1, 100U * \
                                        sizeof(float) << 12, cudaMemcpyHostToDevice, s1));
                CHECK_CUDA_ERROR(cudaMemcpyAsync(d_t_weight_1, t_weight_1, 9U * \
                                        sizeof(float) << 8,  cudaMemcpyHostToDevice, s2));

		dim3 block_size_0(32, 8);
		dim3 grid_size_0(160, 1);
		dim3 block_size_1(16, 16);
		dim3 grid_size_1(9, 1);
		dim3 block_size_2(16, 16);
                dim3 grid_size_2(800, 1);

		// Warm up run
		Convm1_input_reshape_25600x16_25600x144<float>
						       <<<grid_size_0, block_size_0, 0, s1>>>
						       (d_t_input_1, reshaped_input);
		CHECK_LAST_CUDA_ERROR();

		Convm1_weight_reshape_16x16x3x3_144x16<float>
						      <<<grid_size_1, block_size_1, 0, s2>>>
						      (d_t_weight_1, reshaped_weight);
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventRecord(e1, s1));
		CHECK_CUDA_ERROR(cudaEventRecord(e2, s2));
		CHECK_CUDA_ERROR(cudaStreamWaitEvent(s3, e2, 0));
		CHECK_CUDA_ERROR(cudaStreamWaitEvent(s3, e1, 0));


		Convm1_25600x144x16_SiLU_adding<float>
					<<<grid_size_2, block_size_2>>>
					(reshaped_input, reshaped_weight, d_t_bias_1, nullptr, d_t_output_1);
                CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		

		// Timed run
		CHECK_CUDA_ERROR(cudaEventRecord(start, ctrl));
		CHECK_CUDA_ERROR(cudaStreamWaitEvent(s1, start, 0));
		CHECK_CUDA_ERROR(cudaStreamWaitEvent(s2, start, 0));

		Convm1_input_reshape_25600x16_25600x144<float>
                                                       <<<grid_size_0, block_size_0, 0, s1>>>
                                                       (d_t_input_1, reshaped_input);
                CHECK_LAST_CUDA_ERROR();

		Convm1_weight_reshape_16x16x3x3_144x16<float>
                                                      <<<grid_size_1, block_size_1, 0, s2>>>
                                                      (d_t_weight_1, reshaped_weight);
                CHECK_LAST_CUDA_ERROR();


		CHECK_CUDA_ERROR(cudaEventRecord(e1, s1));
		CHECK_CUDA_ERROR(cudaEventRecord(e2, s2));
		CHECK_CUDA_ERROR(cudaStreamWaitEvent(s3, e2, 0));
                CHECK_CUDA_ERROR(cudaStreamWaitEvent(s3, e1, 0));
                
		Convm1_25600x144x16_SiLU_adding<float>
                                        <<<grid_size_2, block_size_2>>>
                                        (reshaped_input, reshaped_weight, d_t_bias_1, nullptr, d_t_output_1);
                CHECK_LAST_CUDA_ERROR();
		
		CHECK_CUDA_ERROR(cudaEventRecord(stop, s3));
		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

		CHECK_CUDA_ERROR(cudaEventElapsedTime(&runtime, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(t_output_1, d_t_output_1, 100U * \
                                        sizeof(float) << 12, cudaMemcpyDeviceToHost));

		CPU_Convm1_25600x16_16x16x3x3(t_input_1, t_weight_1, t_bias_1, t_gt_1);

		int false_num = 0;
		bool flag4 = true;
		for(int i = 0; i < 25600; i++){
			for(int j = 0; j < 16; j++){
				int offset = i * 16 + j;
				if(abs(t_output_1[offset] - t_gt_1[offset]) > 0.0001f){
					flag4 = false;
					false_num++;
				}
				/* else{
					std::cout << "correct at row: " << i << ", col: " << j << std::endl;
				} */
			}
		}
		
		CHECK_CUDA_ERROR(cudaFreeHost(t_input_1));
		CHECK_CUDA_ERROR(cudaFreeHost(t_bias_1));
		CHECK_CUDA_ERROR(cudaFreeHost(t_weight_1));

                CHECK_CUDA_ERROR(cudaFree(d_t_input_1));
                CHECK_CUDA_ERROR(cudaFree(d_t_weight_1));
                CHECK_CUDA_ERROR(cudaFree(d_t_bias_1));
                CHECK_CUDA_ERROR(cudaFree(d_t_output_1));
                CHECK_CUDA_ERROR(cudaFree(reshaped_input));
                CHECK_CUDA_ERROR(cudaFree(reshaped_weight));
		
		free(t_gt_1);



		std::cout << "Test 4 passed?: " << flag4 << std::endl;
                std::cout << "Mistake on "<< false_num << " elements out of 409600" << std::endl;
                std::cout << "Test 4 runtime: " << runtime * 1000 << " us" << std::endl;

		std::cout << "Unit Test 4 on 3x3 Conv done." << std::endl
                        << std::endl;
		std::cout << std::endl;
        }

	if(test_flags[4]){

                // Test for ? with Gelu activation
		std::cout << "-----------------------------------------------" << std::endl;
                std::cout << "Unit Test 5 on ? begins." << std::endl;
                std::cout << "-----------------------------------------------"
                          << std::endl;

                std::cout << "Unit Test 5 on ? done." << std::endl
                        << std::endl;
		std::cout << std::endl;
        }

	CHECK_CUDA_ERROR(cudaEventDestroy(start));
    	CHECK_CUDA_ERROR(cudaEventDestroy(stop));
  }
	CHECK_CUDA_ERROR(cudaDeviceReset());
	return 0;
}

