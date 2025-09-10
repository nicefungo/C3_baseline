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
	//TODO
	return;
}

template<typename T>
__global__ void Convm1_trivial(const T * input, const T * weight, const T * \
                bias, T * D, unsigned int offset)
{
	TODO();
}



template<typename T>
void CPU_Conv_3200x16x32_SiLU(const T * input, const T * weight, const T * bias, \
		T * D)
{
	for (int i = 0; i < 3200 * 16; ++i) {
        	D[i] = 0;
    	}
	for(int m = 0; m < 3200; m++){
		for(int n = 0; n < 16; n++){
			for(int k = 0; k < 32; k++){
				D[m * 16 + n] += input[m * 32 + k] \
						 * weight[k * 16 + n];
			}
			
			D[m * 16 + n] += bias[n];
			// TODO: SiLU
			D[m * 16 + n] = silu<float>(D[m * 16 + n]);
			
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

	__shared__ T tiled_input[16][33];
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
	
	
	__shared__ T tiled_weight[32][17];
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

	// TODO: SiLU;


	D[(start_pos >> 1) + (row << 4)+ col] = sum;

	return;
}



template<typename T>
__global__ void Conv_3200x32x32_SiLU(const T * input, const T * weight, \
                const T * bias, T * D, unsigned int offset)
{
	// 
}

template<typename T>
void C3(const T * input, const T * weights, const T * biases, T * D, T * buffer){
	
	// kernel size dummy
	int gridsize_dummy = 1, blocksize_dummy = 1;

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
		std::cout << "Unit tests begin" << std::endl
                	<< "-----------------------------------------------"
                	<< std::endl
                	<< "-----------------------------------------------"
                	<< std::endl
			<< std::endl;
	if(test_flags[0]){	
		// Test for Conv 3200 * 32 * 32 with Gelu activation
		 
		std::cout << "Unit Test 1 on Conv3 begins." << std::endl;
		std::cout << "-----------------------------------------------"
			  << std::endl;

		std::cout << "Unit Test 1 on Conv3 done." << std::endl
                        << std::endl;
		
	}

	if(test_flags[1]){
		// Test for Conv 3200 * 16 * 32 with Gelu activation
		std::cout << "Unit Test 2 on Conv1 begins." << std::endl;
		std::cout << "-----------------------------------------------"
			  << std::endl;

		float * t_input_2, * t_weight_2, * t_bias_2, * t_output_2;

		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_input_2, 100U * sizeof(float) << 10, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_weight_2, sizeof(float) << 9, cudaHostAllocDefault));
		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_bias_2, sizeof(float) << 4, cudaHostAllocDefault));

		

		CHECK_CUDA_ERROR(cudaHostAlloc((void**)&t_output_2, 100U * sizeof(float) << 9, cudaHostAllocDefault));
		float * t_gt_2 = (float *)malloc(100U * sizeof(float) << 9);
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

		float * d_t_input_2, * d_t_weight_2, * d_t_bias_2, * d_t_output_2;
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


		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaMemcpy(t_output_2, d_t_output_2, 100U * \
					sizeof(float) << 9, cudaMemcpyDeviceToHost));

		CPU_Conv_3200x16x32_SiLU(t_input_2, t_weight_2, t_bias_2, t_gt_2);	

		bool flag2 = true;
		int false_num = 0;
		for(int i = 0; i < 3200; i++){
			for(int j = 0; j < 16; j++){
				if(i >= 0 && i <= 3 && j >= 0 && j <= 3){
					std::cout << t_gt_2[i * 16 + j] << " " << t_output_2[i * 16 + j] << std::endl;
				}
				if(abs(t_gt_2[i * 16 + j] - t_output_2[i * 16 + j]) > 0.001f){
					flag2 = false;
					false_num++;
				}
			}
		}
		std::cout << "Test 2 passed?: " << flag2 << std::endl;
		std::cout << "Mistake on "<< false_num << " elements out of 51200" << std::endl;

		
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
	}


	if(test_flags[2]){

		// Test for Fused Conv 3200 * 32 * 32 with Gelu activation
		// and Conv 3200 * 16 * 16 with Gelu activation
		std::cout << "Unit Test 3 on fused layers begins." << std::endl;
                std::cout << "-----------------------------------------------"
                          << std::endl;	

		std::cout << "Unit Test 3 on fused layers done." << std::endl
                        << std::endl;
	}

	if(test_flags[3]){

                // Test for 3x3 kernel Conv 3200 * 16 * 16 with Gelu activation
                std::cout << "Unit Test 4 on 3x3 Conv begins." << std::endl;
                std::cout << "-----------------------------------------------"
                          << std::endl;

                std::cout << "Unit Test 4 on 3x3 Conv done." << std::endl
                        << std::endl;
        }

	if(test_flags[4]){

                // Test for ? with Gelu activation
                std::cout << "Unit Test 5 on ? begins." << std::endl;
                std::cout << "-----------------------------------------------"
                          << std::endl;

                std::cout << "Unit Test 5 on ? done." << std::endl
                        << std::endl;
        }

	


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

