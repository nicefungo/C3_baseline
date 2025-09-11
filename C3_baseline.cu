#include "C3_baseline.h"
#include "util.h"

void TODO(){
	std::cout << "TODO stuff undone in `cuda_baseline.cu`." << std::endl;
	std::exit(EXIT_FAILURE);
}

template<typename T>
__global__ void fused_3200x16x32_3200x16x16_SiLU(const T * input, const T * \
                Conv1_weight, const T * Conv1_bias, const T * Convm0_weight,\
                const T * Convm0_bias, T * D1, T * D2, unsigned int offset)
{
	//
	// Probably won't need extra global memory
	// Shared A_tile is of size 32 * 32

	__shared__ T A_tile[32][32];

	

	return;
}

template<typename T>
__global__ void Convm1_trivial(const T * input, const T * weight, const T * \
                bias, T * D, unsigned int offset)
{
	// Trivial convolution
	// 
	// each offset compute 20 * 160 in output
	// 
	// Input size: 3600 * ? TODO: deside how to devide
	// 
	//
	// 3x3 kernel size, (1, 1) padding 
	//


	TODO();
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
__global__ void Conv_3200x16x32_SiLU(const T * input, const T * weight, \
                const T * bias, T * D, unsigned int offset)
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

	unsigned int start_pos = (offset * 3200U) << 5;

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
	tiled_input[thread_linear >> 5][thread_linear & 31U] = input[start_pos
                                                    + (blockIdx.y << 9)
                                                    + thread_linear];
        tiled_input[(thread_linear >> 5) + 8][thread_linear & 31U]
						            = input[start_pos
                                                    + (blockIdx.y << 9)
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
	D[(start_pos >> 1) + (row << 4)+ col] = sum;

	return;
}



template<typename T>
__global__ void Conv_3200x32x32_SiLU(const T * input, const T * weight, \
                const T * bias, T * D, unsigned int offset)
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
		
	unsigned int start_pos = (offset * 3200U) << 5;

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

	A_tile[warp_id][lane_id] = input[start_pos + (blockIdx.y << 10)
						    + thread_linear];
	A_tile[warp_id + 8][lane_id] = input[start_pos + (blockIdx.y << 10)
						        + thread_linear + 256];
	A_tile[warp_id + 16][lane_id] = input[start_pos + (blockIdx.y << 10)
                                                        + thread_linear + 512];
	A_tile[warp_id + 24][lane_id] = input[start_pos + (blockIdx.y << 10)
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
	D[start_pos + (blockIdx.y << 10) + (tile_row_start << 5)
					      + tile_col_start] = sum00;

	D[start_pos + (blockIdx.y << 10) + (tile_row_start << 5)
					      + tile_col_start + 1] = sum01;
	D[start_pos + (blockIdx.y << 10) + ((tile_row_start + 1) << 5)
					      + tile_col_start] = sum10;
	D[start_pos + (blockIdx.y << 10) + ((tile_row_start + 1) << 5)
					      + tile_col_start + 1] = sum11;
}

template<typename T>
void C3(const T * input, const T * weights, const T * biases, T * D, T * buffer){
	
	// kernel size dummy
	int gridsize_dummy = 1, blocksize_dummy = 1;
	dim3 block_size(16, 16);

	// Part_1
	cudaStream_t streams1[8];
	cudaStream_t streams2[8];
	//cudaStream_t streams3[16];
	//cudaStream_t streams4[16];

	for(int i = 0; i < 8; i++){
		cudaStreamCreate(&streams1[i]);
	}

	for(int i = 0; i < 8; i++){
		fused_3200x16x32_3200x16x16_SiLU<T>
			<<<gridsize_dummy, blocksize_dummy, 0, streams1[i]>>>
			(input,
		 	weights + CONV_WEIGHT_1_OFFSET, 
		 	biases + CONV_BIAS_1_OFFSET, 
		 	weights + CONV_WEIGHT_m0_OFFSET, 
		 	biases + CONV_BIAS_m0_OFFSET, 
		 	D,
			buffer,
			i
		 	);
	}

	for(int i = 0; i < 8; i++){
		cudaStreamCreate(&streams2[i]);
	}

	for(int i = 0; i < 8; i++){
		std::cout << i << "-th iteration of Conv1 beginning." << std::endl;
		Conv_3200x16x32_SiLU<T>
			<<<gridsize_dummy, blocksize_dummy, 0, streams2[i]>>>
			(input,
			 weights + CONV_WEIGHT_2_OFFSET,
			 biases + CONV_BIAS_2_OFFSET,
			 buffer,
			 i
			 );
	}

	// Temp code for freeing resources in main
	// TODO: remove after finishing the following part
	for(int i = 0; i < 8; i++){
		CHECK_CUDA_ERROR(cudaStreamSynchronize(streams2[i]));
		std::cout << i << "-th iteration of Conv1 finished." << std::endl;
		CHECK_CUDA_ERROR(cudaStreamDestroy(streams2[i]));
	}

	for(int i = 0; i < 8; i++){
		CHECK_CUDA_ERROR(cudaStreamSynchronize(streams1[i]));
		CHECK_CUDA_ERROR(cudaStreamDestroy(streams1[i]));
	}

	// Part_2
}

int main(int arg, char ** args){
	bool test_flags[5];
	if(arg>=2){
		for(int i = 0; i < 5; i++){
			test_flags[i] = (args[1][i]=='1');
		}
	
	}

	float * input, * weights, * biases, * output, * buffer;
	CHECK_CUDA_ERROR(cudaHostAlloc((void**)&input, INPUT_SIZE * sizeof(float), cudaHostAllocDefault));
	CHECK_CUDA_ERROR(cudaHostAlloc((void**)&weights, CONV_WEIGHT_SIZE * sizeof(float), cudaHostAllocDefault));
	CHECK_CUDA_ERROR(cudaHostAlloc((void**)&biases, CONV_BIAS_SIZE * sizeof(float), cudaHostAllocDefault));
	CHECK_CUDA_ERROR(cudaHostAlloc((void**)&output, OUTPUT_SIZE * sizeof(float), cudaHostAllocDefault));

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
	// float * buffer = (float *)malloc(OUTPUT_SIZE * sizeof(float)); 
	CHECK_CUDA_ERROR(cudaHostAlloc((void**)&buffer, OUTPUT_SIZE * sizeof(float), cudaHostAllocDefault));


	float * d_input, * d_weights, * d_biases, * d_output, * d_buffer;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights, CONV_WEIGHT_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_biases, CONV_BIAS_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_buffer, OUTPUT_SIZE * sizeof(float)));


	

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

		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_input_1, 100U * sizeof(float) << 10, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_weight_1, sizeof(float) << 10, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_bias_1, sizeof(float) << 5, cudaHostAllocDefault));

		

		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_output_1, 100U * sizeof(float) << 10, cudaHostAllocDefault));
		t_gt_1 = (float *)malloc(100U * sizeof(float) << 10);
		memset((void*) t_gt_1, 0, 100U * sizeof(float) << 10);

		for(int i = 0; i < 102400; i++){
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

		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_input_1, 100U * sizeof(float) << 10));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_weight_1, sizeof(float) << 10));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_bias_1, sizeof(float) << 5));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_output_1, 100U * sizeof(float) << 10));
			
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_input_1, t_input_1, 100U * \ 
					sizeof(float) << 10, cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_weight_1, t_weight_1, \
					sizeof(float) << 10,  cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_bias_1, t_bias_1, \ 
                                        sizeof(float) << 5, cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemset((void*)d_t_output_1, 0, 100U * sizeof(float) << 10));

		dim3 block_size_1(32, 8);
		dim3 grid_size_1(1, 100);

		Conv_3200x32x32_SiLU<float>
                        <<<grid_size_1, block_size_1>>>
                                (d_t_input_1, \
                                 d_t_weight_1, \
                                 d_t_bias_1, \
                                 d_t_output_1, 0);
		Conv_3200x32x32_SiLU<float>
                        <<<grid_size_1, block_size_1>>>
                                (d_t_input_1, \
                                 d_t_weight_1, \
                                 d_t_bias_1, \
                                 d_t_output_1, 0);
		// Timed run
		cudaEventRecord(start);
		Conv_3200x32x32_SiLU<float>
			<<<grid_size_1, block_size_1>>>
				(d_t_input_1, \
				 d_t_weight_1, \
				 d_t_bias_1, \
			         d_t_output_1, 0);
		cudaEventRecord(stop);	
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&runtime, start, stop);

		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaMemcpy(t_output_1, d_t_output_1, 100U * \
					sizeof(float) << 10, cudaMemcpyDeviceToHost));

		CPU_Conv_MxNxK_SiLU(t_input_1, t_weight_1, t_bias_1, t_gt_1, 3200, 32, 32);	

		bool flag1 = true;
		int false_num = 0;
		for(int i = 0; i < 3200; i++){
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
		std::cout << "Mistake on "<< false_num << " elements out of 102400" << std::endl;
		std::cout << "Test 1 runtime: " << runtime * 1000 << " ns" << std::endl;
		
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
		// Test for Conv 3200 * 16 * 32 with Gelu activation
		std::cout << "-----------------------------------------------" << std::endl;
		std::cout << "Unit Test 2 on Conv1 begins." << std::endl;
		std::cout << "-----------------------------------------------"
			  << std::endl;


		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_input_2, 100U * sizeof(float) << 10, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_weight_2, sizeof(float) << 9, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_bias_2, sizeof(float) << 4, cudaHostAllocDefault));

		

		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_output_2, 100U * sizeof(float) << 9, cudaHostAllocDefault));
		t_gt_2 = (float *)malloc(100U * sizeof(float) << 9);
		memset((void*) t_gt_2, 0, 100U * sizeof(float) << 9);

		for(int i = 0; i < 102400; i++){
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

		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_input_2, 100U * sizeof(float) << 10));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_weight_2, sizeof(float) << 9));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_bias_2, sizeof(float) << 4));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_output_2, 100U * sizeof(float) << 9));
			
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_input_2, t_input_2, 100U * \ 
					sizeof(float) << 10, cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_weight_2, t_weight_2, \
					sizeof(float) << 9,  cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpy(d_t_bias_2, t_bias_2, \ 
                                        sizeof(float) << 4, cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemset((void*)d_t_output_2, 0, 100U * sizeof(float) << 9));

		dim3 block_size_2(16, 16);
		dim3 grid_size_2(1, 200);
		Conv_3200x16x32_SiLU<float>
			<<<grid_size_2, block_size_2>>>
				(d_t_input_2, \
				 d_t_weight_2, \
				 d_t_bias_2, \
			         d_t_output_2, 0);
		Conv_3200x16x32_SiLU<float>
                        <<<grid_size_2, block_size_2>>>
                                (d_t_input_2, \
                                 d_t_weight_2, \
                                 d_t_bias_2, \
                                 d_t_output_2, 0);

		// Timed run
		cudaEventRecord(start);
		Conv_3200x16x32_SiLU<float>
                        <<<grid_size_2, block_size_2>>>
                                (d_t_input_2, \
                                 d_t_weight_2, \
                                 d_t_bias_2, \
                                 d_t_output_2, 0);
		cudaEventRecord(stop);
    		cudaEventSynchronize(stop);
    		cudaEventElapsedTime(&runtime, start, stop);


		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaMemcpy(t_output_2, d_t_output_2, 100U * \
					sizeof(float) << 9, cudaMemcpyDeviceToHost));

		CPU_Conv_MxNxK_SiLU(t_input_2, t_weight_2, t_bias_2, t_gt_2, 3200, 16, 32);	

		bool flag2 = true;
		int false_num = 0;
		for(int i = 0; i < 3200; i++){
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
		std::cout << "Mistake on "<< false_num << " elements out of 51200" << std::endl;
		std::cout << "Test 2 runtime: " << runtime * 1000 << " ns" << std::endl;
		
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

	cudaEventDestroy(start);
    	cudaEventDestroy(stop);

	


	// Init or IO with ONNX exported files
	// TODO


	// TODO: finish C3 Test
	// TODO: There are unsolved illegal mem accesses in Part_1 C3
	// C3<float>(input, weights, biases, output, buffer);
	
	CHECK_LAST_CUDA_ERROR();
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	CHECK_CUDA_ERROR(cudaFree(d_input));
	CHECK_CUDA_ERROR(cudaFree(d_weights));
	CHECK_CUDA_ERROR(cudaFree(d_biases));
	CHECK_CUDA_ERROR(cudaFree(d_output));
	CHECK_CUDA_ERROR(cudaFree(d_buffer));

	CHECK_CUDA_ERROR(cudaFreeHost(input));
	CHECK_CUDA_ERROR(cudaFreeHost(weights));
	CHECK_CUDA_ERROR(cudaFreeHost(biases));
	CHECK_CUDA_ERROR(cudaFreeHost(output));
	CHECK_CUDA_ERROR(cudaFreeHost(buffer));
	
	// Use pinned memory instead
	// free(input); free(weights); free(biases); free(output); free(buffer);
	
	CHECK_CUDA_ERROR(cudaDeviceReset());
	return 0;
}

