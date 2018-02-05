#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "MnistPreProcess.h"
#include <curand.h>
#include <curand_kernel.h>
#include <string.h>

#define FEAT_KEY 0
#define CUT_KEY 1
#define LEFT_KEY 2
#define RIGHT_KEY 3
#define PRED_KEY 4
#define DEPTH_KEY 5

#define NUM_FIELDS 6

#define index(i, j, N)  ((i)*(N)) + (j)
#define index(i, j, N)  ((i)*(N)) + (j)
#define ixt(i, j, t, N, T) ((t)*(N)*(T)) + ((i)*(N)) + (j)
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
 
int countNumRows(char *filename)
{
	FILE *fp;
	int count = 0;  // Line counter (result)
	//char filename[MAX_FILE_NAME];
	char c;  // To store a character read from file
 
	// Get file name from user. The file should be
	// either in current folder or complete path should be provided
	//printf("Enter file name: ");
	//scanf("%s", filename);
 
	// Open the file
	fp = fopen(filename, "r");
 
	// Check if file exists
	if (fp == NULL)
	{
		printf("Could not open file %s", filename);
		return -1;
	}
 
	// Extract characters from file and store in character c
	for (c = getc(fp); c != EOF; c = getc(fp))
		if (c == '\n') // Increment count if this character is newline
			count = count + 1;
 
	// Close the file
	fclose(fp);
	//printf("The file %s has %d lines\n ", filename, count);
 
	return count;
}

const char* getfield(char* line, int num){
	const char* tok;
	for (tok = strtok(line, ",");
			tok && *tok;
			tok = strtok(NULL, ",\n"))
	{
		if (!--num)
			return tok;
	}
	return NULL;
}
/*
Labels for IRIS:
Iris-setosa - 0
Iris-versicolor - 1
Iris-virginica - 2
*/
void read_csv_iris(float *data, float *label, int row_count, char *filename){
	//data = (float *)malloc(row_count*4*sizeof(float));
	//label = (int *)malloc(row_count*sizeof(int));
	FILE *fp = fopen(filename,"r");
	char line[1024];
	int idx = 0;
	for(int iter = 0;iter<row_count;iter++)
	{
		fgets(line,1024,fp);
		const char *temp_field;
		for(int i=0;i<5;i++)
		{
			float temp_num;
			char *tmp = strdup(line);
			temp_field = getfield(tmp,i+1);
			if(i==4)
			{
				if(strcmp(temp_field,"Iris-setosa")==0)
				{
					label[idx] = 0;
					continue;
				}
				if(strcmp(temp_field,"Iris-versicolor")==0)
				{
					label[idx] = 1;
					continue;
				}
				if(strcmp(temp_field,"Iris-virginica")==0)
				{
					label[idx] = 2;
					continue;
				}
			}
			temp_num = atof(temp_field);
			data[idx*4 + i] = temp_num;
		}
		idx++;
		
	}
} 




void readData(float* dataset,float*labels,const char* dataPath,const char*labelPath)
{
	FILE* dataFile=fopen(dataPath,"rb");
	FILE* labelFile=fopen(labelPath,"rb");
	int mbs=0,number=0,col=0,row=0;
	fread(&mbs,4,1,dataFile);
	fread(&number,4,1,dataFile);
	fread(&row,4,1,dataFile);
	fread(&col,4,1,dataFile);
	revertInt(&mbs);
	revertInt(&number);
	revertInt(&row);
	revertInt(&col);
	fread(&mbs,4,1,labelFile);
	fread(&number,4,1,labelFile);
	revertInt(&mbs);
	revertInt(&number);
	unsigned char temp;
	for(int i=0;i<number;++i)
	{
		for(int j=0;j<row*col;++j)
		{
			fread(&temp,1,1,dataFile);
			//dataset[i][j]=static_cast<float>(temp);
			dataset[(i*row*col) + j] = (float)temp;
		}
		fread(&temp,1,1,labelFile);
		//printf("%s\n",*temp );
		//labels[i]=static_cast<float>(temp);
		labels[i] = (float)temp;
		//printf("%f\n", labels[i]);
	}
	fclose(dataFile);
	fclose(labelFile);
}


/* === Utils === */
int next_pow_2(int x){
	int y = 1;
	while(y < x)
		y*=2;
	return y;
}
void copy_transpose(float* to, float* from, int h, int w){
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			to[index(j, i, h)] = from[index(i, j, w)];
		}
	}
}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	// From https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-
	//   to-check-for-errors-using-the-cuda-runtime-api
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
void debug(){
	cudaError_t code;
	code = cudaPeekAtLastError();
	if(code != cudaSuccess){
		printf("GPUassert: Failed at Init: %s\n", cudaGetErrorString(code));
		exit(code);
	}
	code = cudaDeviceSynchronize();
	if(code != cudaSuccess){
		printf("GPUassert: Failed at Execution: %s\n", cudaGetErrorString(code));
		exit(code);
	}
}

/* === Random Init === */
__global__ void init_random(unsigned int seed, curandState_t* states) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, tid, 0, &states[tid]);
}
__device__ int draw_approx_binomial(int n, float p, curandState_t* state) {
	int x = (int) round(curand_normal(state) * n*p*(1-p) + n*p);
	return max(0, min(x, n));
}
__device__ float draw_uniform(float minimum, float maximum, curandState_t* state){
	return minimum + curand_uniform(state) * (maximum - minimum);
}
__device__ int draw_uniform_int(int minimum, int maximum, curandState_t* state){
	return floor(draw_uniform(minimum, maximum, state));
}

/* === Expanding tree memory === */
void expand(float** d_trees_ptr, int num_trees, int tree_arr_length, int new_tree_arr_length){
	float *new_d_trees, *d_trees;
	d_trees = *d_trees_ptr;
	assert(new_tree_arr_length >= tree_arr_length);

	cudaMalloc((void **) &new_d_trees, num_trees * NUM_FIELDS * new_tree_arr_length * sizeof(float));
	for(int i=0; i<num_trees; i++){
		cudaMemcpy(
			new_d_trees + i * (NUM_FIELDS * new_tree_arr_length), 
			d_trees + i * (NUM_FIELDS * tree_arr_length),  
			(NUM_FIELDS * tree_arr_length) * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	cudaFree(d_trees);
	*d_trees_ptr = new_d_trees;
}
__global__ void get_max_tree_length(int* d_tree_lengths, int num_trees, int* d_max_tree_length){
	extern __shared__ int tree_length_buffer[];
	if(threadIdx.x < num_trees){
		tree_length_buffer[threadIdx.x] = d_tree_lengths[threadIdx.x];
	}else{
		tree_length_buffer[threadIdx.x] = -1;
	}
	
	for(int stride=blockDim.x/2; stride > 0; stride >>=1){
		__syncthreads();
		if(threadIdx.x < stride){
			if(tree_length_buffer[threadIdx.x + stride] > tree_length_buffer[threadIdx.x]){
				tree_length_buffer[threadIdx.x] = tree_length_buffer[threadIdx.x + stride];
			}
		}
	}
	if(threadIdx.x == 0){
	   d_max_tree_length[0] = tree_length_buffer[0];
	}
}
void maybe_expand(float** d_trees_ptr, int num_trees, int* tree_arr_length, int* d_tree_lengths,
	                int* max_tree_length, int* d_max_tree_length){
	// I wonder if it's faster just to compute max on CPU.
	int new_tree_arr_length;

	get_max_tree_length<<<1, next_pow_2(num_trees), next_pow_2(num_trees) * sizeof(int)>>>(
		d_tree_lengths, num_trees, d_max_tree_length
	);
	cudaMemcpy(max_tree_length, d_max_tree_length, sizeof(int), cudaMemcpyDeviceToHost);

	// Buffer of 2 => up to 2 additions at a time
	if(*max_tree_length <= *tree_arr_length-4){
		return;
	}else{
		new_tree_arr_length = (*tree_arr_length) * 2;
        while(*max_tree_length > new_tree_arr_length-3){
            new_tree_arr_length *= 2;
        }

        printf("Expanding to %d\n", new_tree_arr_length);
        expand(d_trees_ptr, num_trees, *tree_arr_length, new_tree_arr_length);
        *tree_arr_length = new_tree_arr_length;
	}
}

/* === Tree Initialization === */
__global__ void kernel_initialize_trees(float *d_trees, int* d_tree_lengths, int tree_arr_length){
	d_trees[ixt(0, LEFT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = 0;
	d_trees[ixt(0, RIGHT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = 0;
	d_trees[ixt(0, DEPTH_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = 0;
	d_trees[ixt(0, PRED_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = -1;
	d_tree_lengths[threadIdx.x] = 1;
}
void initialize_trees(float* d_trees, int num_trees, int tree_arr_length, int* d_tree_lengths){
	kernel_initialize_trees<<<1, num_trees>>>(d_trees, d_tree_lengths, tree_arr_length);
}
__global__ void kernel_initialize_batch_pos(int *d_batch_pos, int x_length, int num_trees){
	int i;
	for(i=threadIdx.x; i<x_length; i+=blockDim.x){
		d_batch_pos[index(blockIdx.x, i, x_length)] = 0;
	}
}
void initialize_batch_pos(int *d_batch_pos, int x_length, int num_trees, cudaDeviceProp dev_prop){
	kernel_initialize_batch_pos<<<num_trees, dev_prop.maxThreadsPerBlock>>>(
		d_batch_pos, x_length, num_trees
	);
}

/* === Tree Growth checks === */
__global__ void kernel_refresh_tree_is_done(int* d_tree_lengths, int* d_tree_is_done, int tree_pos){
	// threadIdx.x = tree_id
	int is_done;
	if(tree_pos < d_tree_lengths[threadIdx.x]){
		is_done = 0;
	}else{
		is_done = 1;
	}
	d_tree_is_done[threadIdx.x] = is_done;
}
void refresh_tree_is_done(int* d_tree_lengths, int* d_tree_is_done, int tree_pos, int num_trees){
	kernel_refresh_tree_is_done<<<1, num_trees>>>(
		d_tree_lengths, d_tree_is_done, tree_pos
	);
}
int check_forest_done(int* d_tree_is_done, int *tree_is_done, int num_trees){
	cudaMemcpy(tree_is_done, d_tree_is_done, num_trees * sizeof(int), cudaMemcpyDeviceToHost);
	int trees_left;
	trees_left = 0;
	for(int i=0; i<num_trees; i++){
		if(!tree_is_done[i]){
			trees_left++;
		}
	}
	printf("%d trees left to grow\n", trees_left);
	if(trees_left == 0){
		return 1;
	}else{
		return 0;
	}
}

/* === Tree Traversal === */
__global__ void kernel_traverse_trees(
			float *d_trees, float* d_x, 
			int x_length, int num_trees, int tree_arr_length, 
			int* d_batch_pos, int FEATURE
		){
	// Should optimize this. It's just a bunch of global reads.
	// Also possibly to rewrite this and batch_traverse to support a "next-step" method instead of a full 
	//   traversal while growing
	int pos, new_pos, left_right_key, x_i, tree_id, tx;
	tx = threadIdx.x + blockIdx.x * blockDim.x;
	if(tx >= x_length * num_trees) return;

	// Actually get x_i, tree_id
	tree_id = tx % num_trees;
	x_i = tx / num_trees;
	pos = 0;
    while(1){
        if(d_x[index(x_i, (int) d_trees[
        		ixt(pos, FEAT_KEY, tree_id, NUM_FIELDS, tree_arr_length)], FEATURE)] < 
    				d_trees[ixt(pos, CUT_KEY, tree_id, NUM_FIELDS, tree_arr_length)]){
            left_right_key = LEFT_KEY;
        }else{
            left_right_key = RIGHT_KEY;
        }
        new_pos = (int) d_trees[ixt(pos, left_right_key, tree_id, NUM_FIELDS, tree_arr_length)];
        if(new_pos == pos){
            // Leaf nodes are set up to be idempotent
            break;
        }
        pos = new_pos;
    }
    d_batch_pos[index(tree_id, x_i, x_length)] = pos;
}
void batch_traverse_trees(
			float *d_trees, float *d_x, 
			int x_length, int num_trees, int tree_arr_length, 
			int *d_batch_pos, cudaDeviceProp dev_prop, int FEATURE){
	int block_size, num_blocks;
 	block_size = dev_prop.maxThreadsPerBlock;
 	num_blocks = ceil(num_trees * x_length/((float) block_size));
	kernel_traverse_trees<<<num_blocks, block_size>>>(
		d_trees, d_x, x_length, num_trees, tree_arr_length, d_batch_pos, FEATURE
	);
}
__global__ void kernel_advance_trees(
			float *d_trees, float* d_x, int x_length, int tree_arr_length, 
			int num_trees, int* d_batch_pos, int TRAIN_NUM, int FEATURE
		){
	int pos, left_right_key, x_i;
	// threadIdx.x = x_i, blockIdx.x = tree_id
	for(x_i=threadIdx.x; x_i < x_length; x_i+=blockDim.x){
		pos = d_batch_pos[index(blockIdx.x, x_i, TRAIN_NUM)];
	    if(d_x[index(x_i, (int) d_trees[
	    	ixt(pos, FEAT_KEY, blockIdx.x, NUM_FIELDS, tree_arr_length)], FEATURE)] < 
	    		d_trees[ixt(pos, CUT_KEY, blockIdx.x, NUM_FIELDS, tree_arr_length)]){
	        left_right_key = LEFT_KEY;
	    }else{
	        left_right_key = RIGHT_KEY;
	    }
	    d_batch_pos[index(blockIdx.x, x_i, TRAIN_NUM)] = 
	    	(int) d_trees[ixt(pos, left_right_key, blockIdx.x, NUM_FIELDS, tree_arr_length)];
	}
}
void batch_advance_trees(
			float *d_tree, float *d_x, int x_length, 
			int tree_arr_length, int num_trees, int *d_batch_pos, 
			cudaDeviceProp dev_prop, int TRAIN_NUM, int FEATURE
		){
	kernel_advance_trees<<<num_trees, dev_prop.maxThreadsPerBlock>>>(
		d_tree, d_x, x_length, tree_arr_length, num_trees, d_batch_pos, TRAIN_NUM, FEATURE
	);
}

/* === Node termination === */
__global__ void kernel_check_node_termination(
			float* d_trees, int tree_arr_length,
			float* d_y, int* d_batch_pos, int tree_pos, 
			int* d_is_branch_node, int* d_tree_is_done, int TRAIN_NUM
		){
	// threadIdx.x = tree_id
	int i, base_y, new_y, is_branch_node;

	// If tree is done, it's never a branch node
	if(d_tree_is_done[threadIdx.x]==1){
		d_is_branch_node[threadIdx.x] = 0;
		return;
	}

	// Check for non-unique Y
	base_y = -1;
	is_branch_node = 0;
	for(i=0; i<TRAIN_NUM; i++){
		if(d_batch_pos[index(threadIdx.x, i, TRAIN_NUM)] == tree_pos){
			new_y = d_y[i];
			if(base_y == -1){
				base_y = new_y;
			}else if(base_y != new_y){
				is_branch_node = 1;
				break;
			}
		}
	}
	d_is_branch_node[threadIdx.x] = is_branch_node;

	if(base_y==-1){
		printf("ERROR EMPTY TREE %d\n", threadIdx.x);
		assert(false);
	}

	if(!is_branch_node){
		d_trees[ixt(tree_pos, PRED_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = base_y;
	}
}
void check_node_termination(
			float* d_trees, int tree_arr_length,
			float* d_y, int* d_batch_pos, int tree_pos, 
			int* d_is_branch_node, int* d_tree_is_done,
			int num_trees, int TRAIN_NUM
		){
	kernel_check_node_termination<<<1, num_trees>>>(
		d_trees, tree_arr_length, 
		d_y, d_batch_pos, tree_pos,
		d_is_branch_node, d_tree_is_done, TRAIN_NUM
	);
	debug();
}

/* === Valid features === */
__global__ void kernel_collect_min_max(float* d_x_T, int* d_batch_pos, int desired_pos, int num_trees, 
									   int x_length, float* d_min_max_buffer, int* d_is_branch_node,int TRAIN_NUM,int FEATURE){
	extern __shared__ float shared_min_max[]; // threadIdx.x * 2
	// Ripe for optimization.
	// threadIdx.x = x_i++, blockIdx.x = tree_id, feat = blockIdx.y
	int x_i;
	float minimum, maximum, val;

	if(!d_is_branch_node[blockIdx.x]){
		return;
	}

	minimum = FLT_MAX;
	maximum = -FLT_MAX;
	for(x_i=threadIdx.x; x_i < x_length; x_i+=blockDim.x){
		if(d_batch_pos[index(blockIdx.x, x_i, x_length)] == desired_pos){
			val = d_x_T[index(blockIdx.y, x_i, TRAIN_NUM)];
			if(val < minimum){
				minimum = val;
			}
			if(val > maximum){
				maximum = val;
			}
		}
	}
	shared_min_max[index(threadIdx.x, 0, 2)] = minimum;
	shared_min_max[index(threadIdx.x, 1, 2)] = maximum;

	for(int stride=blockDim.x/2; stride > 0; stride >>=1){
		__syncthreads();
		if(threadIdx.x < stride){
			if(shared_min_max[index(threadIdx.x + stride, 0, 2)] < 
					shared_min_max[index(threadIdx.x, 0, 2)]){
				shared_min_max[index(threadIdx.x, 0, 2)] = 
					shared_min_max[index(threadIdx.x + stride, 0, 2)];
			}
			if(shared_min_max[index(threadIdx.x + stride, 1, 2)] > 
					shared_min_max[index(threadIdx.x, 1, 2)]){
				shared_min_max[index(threadIdx.x, 1, 2)] = 
					shared_min_max[index(threadIdx.x + stride, 1, 2)];
			}
		}
	}
	if(threadIdx.x==0){
		d_min_max_buffer[ixt(blockIdx.y, 0, blockIdx.x, 2, FEATURE)] = shared_min_max[index(0, 0, 2)];
		d_min_max_buffer[ixt(blockIdx.y, 1, blockIdx.x, 2, FEATURE)] = shared_min_max[index(0, 1, 2)];
	}
}
void collect_min_max(float* d_x_T, int* d_batch_pos, int desired_pos, int num_trees, int x_length,
					 float* d_min_max_buffer, int* d_is_branch_node, cudaDeviceProp dev_prop,int TRAIN_NUM,int FEATURE){
	// Ripe for optimization.
	dim3 grid(num_trees, FEATURE);
	kernel_collect_min_max<<<grid, 64, 64 * sizeof(int) * 2>>>(
		d_x_T, d_batch_pos, desired_pos, num_trees, x_length, d_min_max_buffer, d_is_branch_node, TRAIN_NUM, FEATURE
	);	
}
__global__ void kernel_collect_num_valid_feat(
			int* d_num_valid_feat, int* d_random_feats_idx,
			float* d_min_max_buffer, int num_trees, int* d_is_branch_node, int FEATURE
		){
	int feat_i, tree_id, num_valid_feat;
	tree_id = threadIdx.x + blockIdx.x * blockDim.x;
	if(tree_id >= num_trees){
		return;
	}

	num_valid_feat = 0;
	for(feat_i=0; feat_i<FEATURE; feat_i++){
		if(d_min_max_buffer[ixt(feat_i, 0, tree_id, 2, FEATURE)] != 
			d_min_max_buffer[ixt(feat_i, 1, tree_id, 2, FEATURE)]
			){
			d_random_feats_idx[index(tree_id, num_valid_feat, FEATURE)] = feat_i;
			num_valid_feat++;
		}
	}
	d_num_valid_feat[tree_id] = num_valid_feat;
}
void collect_num_valid_feat(
			int* d_num_valid_feat, 
			int* d_random_feats_idx, 
			float* d_min_max_buffer, int num_trees, int* d_is_branch_node, 
			cudaDeviceProp dev_prop, int FEATURE
		){
	// Ripe for optimization
	int grid_size = (int) ceil(1.0 * num_trees / 64);
	int block_size = 64;
	kernel_collect_num_valid_feat<<<grid_size, block_size>>>(
		d_num_valid_feat, d_random_feats_idx,
		d_min_max_buffer, num_trees, d_is_branch_node, FEATURE
	);
}

/* === Populate Random Features === */
__global__ void kernel_populate_feat_cut(
			int* d_random_feats, float* d_random_cuts,
			int* d_random_feats_idx, int* d_num_valid_feat, 
			float* d_min_max_buffer, int feat_per_node,
			int num_trees, int* d_is_branch_node, curandState_t* curand_states, int FEATURE
		){
	int i, num_valid_feat, feat_i, x, tree_id;
	tree_id = threadIdx.x + blockIdx.x * blockDim.x;
	float minimum, maximum;
	if(!d_is_branch_node[tree_id]){
		return;
	}
	num_valid_feat = d_num_valid_feat[tree_id];
	for(i=0; i<feat_per_node; i++){
		x = draw_uniform_int(0, num_valid_feat, curand_states+tree_id);
		feat_i = d_random_feats_idx[index(tree_id, x, FEATURE)];
		minimum = d_min_max_buffer[ixt(feat_i, 0, tree_id, 2, FEATURE)];
		maximum = d_min_max_buffer[ixt(feat_i, 1, tree_id, 2, FEATURE)];
		d_random_feats[index(tree_id, i, feat_per_node)] = feat_i;
		d_random_cuts[index(tree_id, i, feat_per_node)] = 
			draw_uniform(minimum, maximum, curand_states+tree_id);
	}
}
void populate_feat_cut(int* d_random_feats, float* d_random_cuts,
					   int* d_random_feats_idx, int* d_num_valid_feat,
	 				   float* d_min_max_buffer, int feat_per_node,
	 				   int num_trees, int* d_is_branch_node, curandState_t* curand_states,int FEATURE){
	int grid_size = (int) ceil(1.0 * num_trees / 64);
	int block_size = 64;
	kernel_populate_feat_cut<<<grid_size, block_size>>>(
		d_random_feats, d_random_cuts, 
		d_random_feats_idx, d_num_valid_feat, 
		d_min_max_buffer, feat_per_node, num_trees, 
		d_is_branch_node, curand_states, FEATURE
	);
}

/* === Count Classes === */
__global__ void kernel_populate_class_counts(
		float* d_x, float* d_y, int* d_class_counts_a, int* d_class_counts_b, 
		int* d_random_feats, float* d_random_cuts,
		int* d_batch_pos, int tree_pos,
		int num_trees, int feat_per_node, int* d_is_branch_node, int TRAIN_NUM, int NUMBER_OF_CLASSES, int FEATURE
	){
	// Naive version
	// threadIdx.x = tree_id, blockIdx.x = rand_feat_i
	int i, y, feat;
	float cut;
	if(!d_is_branch_node[threadIdx.x]){
		return;
	}
	feat = d_random_feats[index(threadIdx.x, blockIdx.x, feat_per_node)];
	cut = d_random_cuts[index(threadIdx.x, blockIdx.x, feat_per_node)];
	for(i=0; i<NUMBER_OF_CLASSES; i++){
		//tree node class
		d_class_counts_a[ixt(threadIdx.x, blockIdx.x, i, feat_per_node, num_trees)] = 0;
		d_class_counts_b[ixt(threadIdx.x, blockIdx.x, i, feat_per_node, num_trees)] = 0;
	}
	for(i=0; i<TRAIN_NUM; i++){
		if(d_batch_pos[index(threadIdx.x, i, TRAIN_NUM)]==tree_pos){
			y = (int) d_y[i];
			if(d_x[index(i, feat, FEATURE)] < cut){
				d_class_counts_a[ixt(threadIdx.x, blockIdx.x, y, feat_per_node, num_trees)]++;
			}else{
				d_class_counts_b[ixt(threadIdx.x, blockIdx.x, y, feat_per_node, num_trees)]++;
			}
		}
	}
}
void populate_class_counts(
		float* d_x, float* d_y, int* d_class_counts_a, int* d_class_counts_b, 
		int* d_random_feats, float* d_random_cuts,
		int* d_batch_pos, int tree_pos,
		int num_trees, int feat_per_node, int* d_is_branch_node, int TRAIN_NUM, int NUMBER_OF_CLASSES, int FEATURE
	){
	// Naive version
	kernel_populate_class_counts<<<feat_per_node, num_trees>>>(
		d_x, d_y, d_class_counts_a, d_class_counts_b, 
		d_random_feats, d_random_cuts,
		d_batch_pos, tree_pos,
		num_trees, feat_per_node,
		d_is_branch_node, TRAIN_NUM, NUMBER_OF_CLASSES, FEATURE
	);
}

/* === Place Best Features/Cuts === */
__global__ void kernel_place_best_feat_cuts(
		int* d_class_counts_a, int* d_class_counts_b, 
		int* d_random_feats, float* d_random_cuts,
		int* d_best_feats, float* d_best_cuts,
		int feat_per_node, int num_trees, int* d_is_branch_node, int NUMBER_OF_CLASSES
	){
	// Naive version => Can move class_counts into shared memory
	// threadIdx.x = tree_id
	int i, k;
    float best_improvement, best_cut, proxy_improvement;
    int best_feat;
    int total_a, total_b;
    float impurity_a, impurity_b;

	if(!d_is_branch_node[threadIdx.x]){
		return;
	}

    best_improvement = -FLT_MAX;
    best_feat = -1;
    best_cut = 0;
	for(i=0; i<feat_per_node; i++){
        total_a = 0;
        total_b = 0;
        impurity_a = 1;
        impurity_b = 1;
        for(k=0; k<NUMBER_OF_CLASSES; k++){
            total_a += d_class_counts_a[ixt(threadIdx.x, i, k, feat_per_node, num_trees)];
            total_b += d_class_counts_b[ixt(threadIdx.x, i, k, feat_per_node, num_trees)];
        }
        for(k=0; k<NUMBER_OF_CLASSES; k++){
            impurity_a -= pow(((float) d_class_counts_a[
            	ixt(threadIdx.x, i, k, feat_per_node, num_trees)]) / total_a, 2);
            impurity_b -= pow(((float) d_class_counts_b[
            	ixt(threadIdx.x, i, k, feat_per_node, num_trees)]) / total_b, 2);
        }
        proxy_improvement = - total_a * impurity_a - total_b * impurity_b;
        if(proxy_improvement > best_improvement){
            best_feat = d_random_feats[index(threadIdx.x, i, feat_per_node)];
            best_cut = d_random_cuts[index(threadIdx.x, i, feat_per_node)];
            best_improvement = proxy_improvement;
        }
	}
	d_best_feats[threadIdx.x] = best_feat;
	d_best_cuts[threadIdx.x] = best_cut;
}
void place_best_feat_cuts(
		int* d_class_counts_a, int* d_class_counts_b, 
		int* d_random_feats, float* d_random_cuts,
		int* d_best_feats, float* d_best_cuts,
		int feat_per_node, int num_trees, int* d_is_branch_node, int NUMBER_OF_CLASSES
	){
	// Naive version
	kernel_place_best_feat_cuts<<<1, num_trees>>>(
		d_class_counts_a, d_class_counts_b, 
		d_random_feats, d_random_cuts,
		d_best_feats, d_best_cuts,
		feat_per_node, num_trees,
		d_is_branch_node, NUMBER_OF_CLASSES
	);
}

/* === Update Trees === */
__global__ void kernel_update_trees(
			float* d_trees, int* d_tree_lengths, int tree_pos,
			int* d_best_feats, float* d_best_cuts, int tree_arr_length, int* d_is_branch_node
		){
	// Naive version
	// threadIdx.x = tree_id
	int left_child_pos, right_child_pos, tree_length;

	if(!d_is_branch_node[threadIdx.x]){
		return;
	}

	tree_length = d_tree_lengths[threadIdx.x];
	left_child_pos = tree_length;
	right_child_pos = tree_length + 1;

	// Update tree nodes
	d_trees[ixt(tree_pos, LEFT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = left_child_pos;
	d_trees[ixt(tree_pos, RIGHT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = right_child_pos;
	d_trees[ixt(tree_pos, FEAT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = d_best_feats[threadIdx.x];
	d_trees[ixt(tree_pos, CUT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = d_best_cuts[threadIdx.x];
	d_tree_lengths[threadIdx.x] += 2;

	// Prefill child nodes
	d_trees[ixt(left_child_pos, LEFT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = left_child_pos;
	d_trees[ixt(left_child_pos, RIGHT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = left_child_pos;
	d_trees[ixt(left_child_pos, DEPTH_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = \
		d_trees[ixt(tree_pos, DEPTH_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] + 1;
	d_trees[ixt(left_child_pos, FEAT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = -1;
	d_trees[ixt(left_child_pos, CUT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = -1;
	d_trees[ixt(left_child_pos, PRED_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = -1;

	d_trees[ixt(right_child_pos, LEFT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = right_child_pos;
	d_trees[ixt(right_child_pos, RIGHT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = right_child_pos;
	d_trees[ixt(right_child_pos, DEPTH_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = \
		d_trees[ixt(tree_pos, DEPTH_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] + 1;
	d_trees[ixt(right_child_pos, FEAT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = -1;
	d_trees[ixt(right_child_pos, CUT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = -1;
	d_trees[ixt(right_child_pos, PRED_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = -1;
}
void update_trees(
			float* d_trees, int* d_tree_lengths, int tree_pos,
			int* d_best_feats, float* d_best_cuts, int tree_arr_length,
			int num_trees, int* d_is_branch_node
		){
	kernel_update_trees<<<1, num_trees>>>(
		d_trees, d_tree_lengths, tree_pos,
		d_best_feats, d_best_cuts, tree_arr_length, d_is_branch_node
	);
}

/* === Evaluate === */
__global__ void kernel_raw_predict(
			float *d_raw_pred_y, float* d_trees, int* d_batch_pos, int tree_arr_length, int x_length
		){
	// threadIdx.x = tree_id, blockIdx.x = x_i
	int pos;
	pos = d_batch_pos[index(threadIdx.x, blockIdx.x, x_length)];
	d_raw_pred_y[index(threadIdx.x, blockIdx.x, x_length)] = d_trees[
		ixt(pos, PRED_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)];
}
void raw_predict(
			float *d_raw_pred_y, float* d_trees, int* d_batch_pos, int tree_arr_length, int x_length,
			int num_trees
		){
	kernel_raw_predict<<<x_length, num_trees>>>(
		d_raw_pred_y, d_trees, d_batch_pos, tree_arr_length, x_length
	);
}
void predict(float* pred_y, float* raw_pred_y, int x_length, int num_trees, int NUMBER_OF_CLASSES){
	int *class_count_buffer;
	int i, j, k, pred, maximum, maximum_class;
	class_count_buffer = (int *)malloc(NUMBER_OF_CLASSES * sizeof(int));
	for(k=0; k<NUMBER_OF_CLASSES; k++){
		class_count_buffer[k] = 0;
	}
	for(i=0; i<x_length; i++){
		for(j=0; j<num_trees; j++){
			pred = (int) raw_pred_y[index(j, i, x_length)];
			class_count_buffer[pred]++;
		}
		maximum = -1;
		for(k=0; k<NUMBER_OF_CLASSES; k++){
			if(class_count_buffer[k] > maximum){
				maximum = class_count_buffer[k];
				maximum_class = k;
			}
			class_count_buffer[k] = 0;
		}
		pred_y[i] = (float) maximum_class;
	}
}
float evaluate(float* pred_y, float* true_y, int y_length){
	int i;
	float score;
	score = 0;
	for(i=0; i<y_length; i++){
		if((int) pred_y[i] == (int) true_y[i]){
			score += 1;
		}
	}
	score /= y_length;
	return score;
}


int main(int argc, char * argv[]){
	float *dataset_train,*dataset_test;
	float *labels_train,*labels_test;
	int mnist_iris;
	int num_trees;
	int seed;

	if(argc != 4){
		fprintf(stderr, "usage: dataset num_trees seed\n");
		fprintf(stderr, "dataset: MNIST=0, IRIS=1\n");
		fprintf(stderr, "num_trees: <1024\n");
		fprintf(stderr, "seed: int\n");
		exit(1);
	}
	
	mnist_iris = atoi(argv[1]);
	num_trees = atoi(argv[2]);
	seed = atoi(argv[3]);
	srand(seed);

	int TRAIN_NUM,FEATURE,TEST_NUM,NUMBER_OF_CLASSES;
	char file_train_set[50],file_train_label[50],file_test_label[50],file_test_set[50];
	
	if(mnist_iris == 0){
		TRAIN_NUM = 60000;
		TEST_NUM = 10000;
		FEATURE =  784;
		NUMBER_OF_CLASSES = 10;

		dataset_train = (float *)malloc(FEATURE * TRAIN_NUM*sizeof(float));
		labels_train = (float *)malloc(TRAIN_NUM*sizeof(float));
		dataset_test = (float *)malloc(FEATURE * TEST_NUM*sizeof(float));
		labels_test = (float *)malloc(TEST_NUM*sizeof(float));

		strncpy(file_test_set, "data/t10k-images-idx3-ubyte", 50);
		strncpy(file_train_set,"data/train-images-idx3-ubyte",50);
		strncpy(file_train_label, "data/train-labels-idx1-ubyte",50);
		strncpy(file_test_label,"data/t10k-labels-idx1-ubyte",50);

		readData(dataset_train,labels_train,file_train_set,file_train_label);
		readData(dataset_test,labels_test,file_test_set,file_test_label);

	}else if(mnist_iris == 1){
		TRAIN_NUM = 100;
		TEST_NUM = 50;
		FEATURE =  4;
		NUMBER_OF_CLASSES = 3;

		dataset_train = (float *)malloc(FEATURE * TRAIN_NUM*sizeof(float));
		labels_train = (float *)malloc(TRAIN_NUM*sizeof(float));
		dataset_test = (float *)malloc(FEATURE * TEST_NUM*sizeof(float));
		labels_test = (float *)malloc(TEST_NUM*sizeof(float));
		strncpy(file_train_set, "data/iris_train.data",50);
		strncpy(file_test_set,"data/iris_test.data",50);
		read_csv_iris(dataset_train,labels_train,TRAIN_NUM,file_train_set);
		read_csv_iris(dataset_test,labels_test,TEST_NUM,file_test_set);
	}

	float *dataset_train_T;
	dataset_train_T = (float *)malloc(TRAIN_NUM * FEATURE * sizeof(float));
	copy_transpose(dataset_train_T, dataset_train, TRAIN_NUM, FEATURE);

	float *d_trees;
	int *tree_arr_length;
	int *d_tree_lengths;
	int *max_tree_length, *d_max_tree_length;
	int feat_per_node;
	int *d_num_valid_feat;
	int tree_pos;
	int *batch_pos, *d_batch_pos; // NUM_TREES * TRAIN_NUM
	int *d_is_branch_node;
	int *tree_is_done, *d_tree_is_done;
	float *d_min_max_buffer;
	int *d_random_feats_idx;
	int *d_random_feats;
	float *d_random_cuts;
	int *d_class_counts_a, *d_class_counts_b;
	int *d_best_feats;
	float *d_best_cuts;
	float *d_x, *d_y;
	float *d_x_T;
	float *pred_y, *raw_pred_y, *d_raw_pred_y;
	curandState_t* curand_states;

	// Assumption: num_trees < maxNumBlocks, maxThreadsPerBlock

	tree_arr_length = (int *)malloc(sizeof(int));
	*tree_arr_length = 8;
	max_tree_length = (int *)malloc(sizeof(int));

	feat_per_node = (int) ceil(sqrt(FEATURE));

	batch_pos = (int *)malloc(num_trees * TRAIN_NUM *sizeof(float));
	tree_is_done = (int *)malloc(num_trees * sizeof(int));
	
	cudaDeviceProp dev_prop;
	cudaGetDeviceProperties(&dev_prop, 0);
	cudaMalloc((void **) &d_trees, num_trees * NUM_FIELDS * (*tree_arr_length) *sizeof(float));
	cudaMalloc((void **) &d_tree_lengths, num_trees * sizeof(int));
	cudaMalloc((void **) &d_max_tree_length, sizeof(int));
	cudaMalloc((void **) &d_batch_pos, num_trees * TRAIN_NUM *sizeof(float));
	cudaMalloc((void **) &d_is_branch_node, num_trees * sizeof(int));
	cudaMalloc((void **) &d_tree_is_done, num_trees * sizeof(int));
	cudaMalloc((void **) &d_min_max_buffer, num_trees * FEATURE * 2 *sizeof(float));
	cudaMalloc((void **) &d_num_valid_feat, num_trees *sizeof(int));
	cudaMalloc((void **) &d_random_feats_idx, num_trees * FEATURE * sizeof(int));
	cudaMalloc((void **) &d_random_feats, num_trees * feat_per_node * sizeof(int));
	cudaMalloc((void **) &d_random_cuts, num_trees * feat_per_node * sizeof(float));
	cudaMalloc((void **) &d_best_feats, num_trees * sizeof(int));
	cudaMalloc((void **) &d_best_cuts, num_trees * sizeof(float));
	cudaMalloc((void **) &d_class_counts_a, num_trees * feat_per_node * NUMBER_OF_CLASSES *sizeof(int));
	cudaMalloc((void **) &d_class_counts_b, num_trees * feat_per_node * NUMBER_OF_CLASSES *sizeof(int));
	cudaMalloc((void **) &d_x, TRAIN_NUM * FEATURE *sizeof(float));
	cudaMalloc((void **) &d_y, TRAIN_NUM *sizeof(float));
	cudaMalloc((void **) &d_x_T, TRAIN_NUM * FEATURE *sizeof(float));
	cudaMemcpy(d_x, dataset_train, TRAIN_NUM * FEATURE *sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, labels_train, TRAIN_NUM *sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x_T, dataset_train_T, TRAIN_NUM * FEATURE *sizeof(float), cudaMemcpyHostToDevice);


	cudaMalloc((void**) &curand_states, num_trees * sizeof(curandState));
	init_random<<<1, num_trees>>>(1337, curand_states);

	initialize_trees(d_trees, num_trees, *tree_arr_length, d_tree_lengths);
	initialize_batch_pos(d_batch_pos, TRAIN_NUM, num_trees, dev_prop);

	for(tree_pos=0; tree_pos<100000; tree_pos++){
		printf("* ================== TREE POS -[ %d ]- ================== *\n", tree_pos);

		refresh_tree_is_done(d_tree_lengths, d_tree_is_done, tree_pos, num_trees);
		if(check_forest_done(d_tree_is_done, tree_is_done, num_trees)){
			printf("DONE\n");
			break;
		}

		maybe_expand(
			&d_trees, num_trees, tree_arr_length, d_tree_lengths, max_tree_length, d_max_tree_length);

		batch_advance_trees(d_trees, d_x, TRAIN_NUM, *tree_arr_length, num_trees, d_batch_pos, dev_prop, TRAIN_NUM, FEATURE);

		check_node_termination(
			d_trees, *tree_arr_length, 
			d_y, d_batch_pos, tree_pos,
			d_is_branch_node, d_tree_is_done,
			num_trees, TRAIN_NUM
		);
		collect_min_max(
			d_x_T, d_batch_pos, tree_pos, num_trees, TRAIN_NUM,
			d_min_max_buffer, d_is_branch_node, dev_prop, TRAIN_NUM, FEATURE
		);
		collect_num_valid_feat(
			d_num_valid_feat, d_random_feats_idx,
			d_min_max_buffer, num_trees, d_is_branch_node, dev_prop, FEATURE
		);
		populate_feat_cut(
			d_random_feats, d_random_cuts, 
			d_random_feats_idx, d_num_valid_feat, 
			d_min_max_buffer, feat_per_node, num_trees, 
			d_is_branch_node, curand_states, FEATURE
		);
		populate_class_counts(
			d_x, d_y, d_class_counts_a, d_class_counts_b, 
			d_random_feats, d_random_cuts,
			d_batch_pos, tree_pos,
			num_trees, feat_per_node, 
			d_is_branch_node, TRAIN_NUM, NUMBER_OF_CLASSES, FEATURE
		);
		place_best_feat_cuts(
			d_class_counts_a, d_class_counts_b, 
			d_random_feats, d_random_cuts,
			d_best_feats, d_best_cuts,
			feat_per_node, num_trees, 
			d_is_branch_node, NUMBER_OF_CLASSES
		);
		update_trees(
			d_trees, d_tree_lengths, tree_pos,
			d_best_feats, d_best_cuts, *tree_arr_length,
			num_trees, 
			d_is_branch_node
		);
		cudaDeviceSynchronize();
	}

	printf("================= DONE TRAINING =================\n");
	/* === TEST === */
	cudaFree(d_batch_pos);
	free(batch_pos);
	cudaMalloc((void **) &d_batch_pos, num_trees * TEST_NUM * sizeof(float));
	pred_y = (float *)malloc(TEST_NUM * sizeof(float));
	raw_pred_y = (float *)malloc(num_trees * TEST_NUM * sizeof(float));

	cudaFree(d_x);
	cudaMalloc((void **) &d_x, TEST_NUM * FEATURE * sizeof(float));
	cudaMalloc((void **) &d_raw_pred_y, num_trees * TEST_NUM * sizeof(float));
	cudaMemcpy(d_x, dataset_test, TEST_NUM * FEATURE * sizeof(float), cudaMemcpyHostToDevice);

	initialize_batch_pos(
		d_batch_pos, TEST_NUM, num_trees, dev_prop
	);
	batch_traverse_trees(
		d_trees, d_x, TEST_NUM, num_trees, *tree_arr_length, d_batch_pos, dev_prop, FEATURE
	);
	cudaMemcpy(d_x, dataset_test, TEST_NUM * FEATURE * sizeof(float), cudaMemcpyHostToDevice);
	raw_predict(d_raw_pred_y, d_trees, d_batch_pos, *tree_arr_length, TEST_NUM, num_trees);
	cudaMemcpy(raw_pred_y, d_raw_pred_y, num_trees * TEST_NUM * sizeof(float), cudaMemcpyDeviceToHost);
	predict(pred_y, raw_pred_y, TEST_NUM, num_trees, NUMBER_OF_CLASSES);

	printf("Test Accuracy: %f\n", evaluate(pred_y, labels_test, TEST_NUM));
	debug();
}