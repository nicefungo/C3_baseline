#include "C3_baseline.h"

void TODO(){
	std::cout << "TODO stuff undone in `cuda_baseline.cu`." << std::endl;
	std::exit(EXIT_FAILURE);
}

template<typename T>
__global__ void fused_3200x16x32_3200x16x16_GELU(const T * input, const T * \
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
__global__ void Conv_3200x16x32_GELU(const T * input, const T * weight, \
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
	// M = 3200, N = 32, K = 16
	// Tiling size = M: 16, N: 16, K: 16
	// TODO: try tiling size M: 16, N: 16, K: 8 ?
	// Tiling dim: tild_M = 100, tile_N = 2, tile_K = 1
	



	// default block size 16 * 16
	// each block tile compute a 16 * 16 tile

	unsigned int start_pos = offset * 3200U;

	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	// block tile size 16 * 16
	// block size 16 * 16
	unsigned int block_linear = blockDim.x * blockIdx.y + blockIdx.x;
	unsigned int thread_linear = blockDim.x * threadIdx.y + threadIdx.x;

	// 8 warps per block
	unsigned int warp_id = (thread_linear >> 5);
	unsigned int lane_id = thread_linear & 31U;
	
	
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
	// TODO: Colescing and correct?
	tiled_input[thread_linear >> 5][thread_linear & 31U] = input[start_pos
                                                    + (block_linear << 8)
                                                    + thread_linear];
        tiled_input[(thread_linear + 256U) >> 5][(thread_linear + 256U) & 31U]
						            = input[start_pos
                                                    + (block_linear << 8)
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
	
	unsigned int warp_tile_id_x = warp_id >> 1;
	unsigned int warp_tile_id_y = warp_id & 1;

	//T A_val[][]


	
	






	return;
}

template<typename T>
__global__ void Conv_3200x32x32_GELU(const T * input, const T * weight, \
                const T * bias, T * D, unsigned int offset)
{
	TODO();
}

template<typename T>
void C3(const T * input, const T * weights, const T * biases, T * D, T * temp){
	
	// kernel size dummy
	int gridsize_dummy = 1, blocksize_dummy = 1;

	cudaStream_t streams1[8];
	cudaStream_t streams2[8];
	//cudaStream_t streams3[16];
	//cudaStream_t streams4[16];

	for(int i = 0; i < 8; i++){
		cudaStreamCreate(&streams1[i]);
	}

	for(int i = 0; i < 8; i++){
		fused_3200x16x32_3200x16x16_GELU<T>
			<<<gridsize_dummy, blocksize_dummy, 0, streams1[i]>>>
			(input,
		 	weights + CONV_WEIGHT_1_OFFSET, 
		 	biases + CONV_BIAS_1_OFFSET, 
		 	weights + CONV_WEIGHT_m0_OFFSET, 
		 	biases + CONV_BIAS_m0_OFFSET, 
		 	D,
			temp,
			i
		 	);
	}

	for(int i = 0; i < 8; i++){
		cudaStreamCreate(&streams2[i]);
	}

	for(int i = 0; i < 8; i++){
		Conv_3200x16x32_GELU<T>
			<<<gridsize_dummy, blocksize_dummy, 0, streams2[i]>>>
			(input,
			 weights + CONV_WEIGHT_2_OFFSET,
			 biases + CONV_BIAS_2_OFFSET,
			 temp,
			 i
			 );
	}
}

int main(int arg, char ** args){
	bool test_flag;	
	if(arg>=1){
	
		test_flag = (bool) *args[1];
	
	}

	float * input = (float *)malloc(INPUT_SIZE * sizeof(float));
	float * weights = (float *)malloc(CONV_WEIGHT_SIZE * sizeof(float));
	float * biases = (float *)malloc(CONV_BIAS_SIZE * sizeof(float));
	float * output = (float *)malloc(OUTPUT_SIZE * sizeof(float));

	float * temp = (float *)malloc(OUTPUT_SIZE * sizeof(float)); 

	if(test_flag){
		// Unit tests
		std::cout << "Unit tests begin" << std::endl
                	<< "-----------------------------------------------"
                	<< std::endl;
		
		// Test for Conv 3200 * 32 * 32 with Gelu activation
		



		// Test for Conv 3200 * 16 * 32 with Gelu activation
		



		// Test for Fused Conv 3200 * 32 * 32 with Gelu activation
		// and Conv 3200 * 16 * 16 with Gelu activation
		 
	}


	// Init or IO with ONNX exported files
	// TODO

	C3<float>(input, weights, biases, output, temp);



	free(input); free(weights); free(biases); free(output); free(temp);
	return 0;
}





