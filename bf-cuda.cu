/*
* This is a CUDA version of bellman_ford algorithm
* Compile: nvcc -std=c++11 -arch=sm_52 -o cuda_bellman_ford cuda_bellman_ford.cu
* Run: ./cuda_bellman_ford <input file> <number of blocks per grid> <number of threads per block>, you will find the output file 'output.txt'
* */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <ctime>


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "pnt.hpp"

using std::string;
using std::cout;
using std::endl;

#define INF 1000000


void pprint(int i, int n, bool stay = false)
{
	int p = (i + 1) * 100 / n;
	if (p != i * 100 / n)
		printf("%d%%\r", p);
	if (stay && p == 100)
		putchar('\n');
}

/*
* This is a CHECK function to check CUDA calls
*/
#define CHECK(call)                                                            \
		{                                                                              \
	const cudaError_t error = call;                                            \
	if (error != cudaSuccess)                                                  \
	{                                                                          \
		fprintf(stderr, "Error: %s%d, ", __FILE__, __LINE__);                 \
		fprintf(stderr, "code: %d, reason: %s\n", error,                       \
				cudaGetErrorString(error));                                    \
				exit(1);                                                               \
	}                                                                          \
		}


/**
* utils is a namespace for utility functions
* including I/O (read input file and print results) and matrix dimension convert(2D->1D) function
*/
namespace utils {
	int N; //number of vertices
	int *mat; // the adjacency matrix

	void abort_with_error_message(string msg) {
		std::cerr << msg << endl;
		abort();
	}

	//translate 2-dimension coordinate to 1-dimension
	int convert_dimension_2D_1D(int x, int y, int n) {
		return x * n + y;
	}

	int read_file(string filename) {
		std::ifstream inputf(filename, std::ifstream::in);
		if (!inputf.good()) {
			abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
		}
		inputf >> N;
		//input matrix should be smaller than 20MB * 20MB (400MB, we don't have too much memory for multi-processors)
		assert(N < (1024 * 1024 * 20));
		mat = (int *)malloc(N * N * sizeof(int));
		printf("%d int malloced\n", N * N);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				inputf >> mat[convert_dimension_2D_1D(i, j, N)];
			}
			pprint(i, N);
		}
		return 0;
	}

	int print_result(bool has_negative_cycle, int *dist) {
		std::ofstream outputf("output.txt", std::ofstream::out);
		if (!has_negative_cycle) {
			for (int i = 0; i < N; i++) {
				if (dist[i] > INF)
					dist[i] = INF;
				outputf << dist[i] << '\n';
			}
			outputf.flush();
		}
		else {
			outputf << "FOUND NEGATIVE CYCLE!" << endl;
		}
		outputf.close();
		return 0;
	}
} //namespace utils

 // you may add some helper/kernel functions here.

__global__ void relax_initial(int * d_dist, bool * d_has_negative_cycle, bool * relaxed_last_round, bool * relaxed_this_round, int * relaxed_times, int n)
{
	int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
	int i = bdim * bid + tid;
	int skip = bdim * gdim;
	for (int k = i; k < n; k += skip) {
		d_dist[k] = INF;
		relaxed_last_round[k] = false;
		relaxed_this_round[k] = false;
		relaxed_times[k] = 0;
	}

	if (i == 0) {
		d_dist[0] = 0;
		*d_has_negative_cycle = false;// changed this morning, you forget * last night
		relaxed_last_round[0] = true;
	}
	__syncthreads();
}

__global__ void relax_swap(bool * relaxed_last_round, bool * relaxed_this_round, int n)
{
	int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
	int i = bdim * bid + tid;
	int skip = bdim * gdim;

	for (int j = i; j < n; j += skip) {
		relaxed_last_round[j] = relaxed_this_round[j];
		relaxed_this_round[j] = false;
	}
	__syncthreads();
}

__global__ void bf(int n, int const* d_mat, int * d_dist, bool * d_has_change, bool * d_has_negative_cycle, bool const* relaxed_last_round, bool * relaxed_this_round, int * relaxed_times)
{
	int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
	int i = bdim * bid + tid;
	int skip = bdim * gdim;

	if (i == 0)
		*d_has_change = false;
	__syncthreads();

	bool my_has_change = false;

	for (int v = i; v < n; v += skip) {
		for (int u = 0; u < n; ++u) {
			if (relaxed_last_round[u]) {
				int weight = d_mat[u * n + v];
				if (weight < INF)
					if (d_dist[u] + weight < d_dist[v]) {
						d_dist[v] = d_dist[u] + weight;
						relaxed_times[v] += 1;
						relaxed_this_round[v] = true;
						my_has_change = true;
						if (v == 0 && d_dist[v] < 0)
							*d_has_negative_cycle = true;
						if (relaxed_times[v] == n)
							*d_has_negative_cycle = true;
					}
			}
		}
	}
	if (my_has_change)
		*d_has_change = true;
}

/**
* Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
* @param blockPerGrid number of blocks per grid
* @param threadsPerBlock number of threads per block
* @param n input size
* @param *mat input adjacency matrix
* @param *dist distance array
* @param *has_negative_cycle a bool variable to recode if there are negative cycles
*/
void bellman_ford(int blocksPerGrid, int threadsPerBlock, int n, int *mat, int *dist, bool *has_negative_cycle) {
	//------your code starts from here-----
	dim3 gdim(blocksPerGrid);
	dim3 bdim(threadsPerBlock);

	bool has_change = false;

	int *d_mat, *d_dist;
	bool *d_has_change, *d_has_negative_cycle;
	bool *relaxed_last_round, *relaxed_this_round;
	int *relaxed_times;

	cudaMalloc(&d_mat, n * n * sizeof(int));
	cudaMalloc(&d_dist, n * sizeof(int));
	cudaMalloc(&d_has_change, sizeof(bool));
	cudaMalloc(&d_has_negative_cycle, sizeof(bool));
	cudaMalloc(&relaxed_last_round, n * sizeof(bool));
	cudaMalloc(&relaxed_this_round, n * sizeof(bool));
	cudaMalloc(&relaxed_times, n * sizeof(int));

	cudaMemcpy(d_mat, mat, n * n * sizeof(int), cudaMemcpyHostToDevice);
		
	relax_initial <<<gdim, bdim>>>(d_dist, d_has_negative_cycle, relaxed_last_round, relaxed_this_round, relaxed_times, n);

	while (true) {
		bf <<<gdim, bdim>>> (n, d_mat, d_dist, d_has_change, d_has_negative_cycle, relaxed_last_round, relaxed_this_round, relaxed_times);
		cudaMemcpy(&has_change, d_has_change, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(has_negative_cycle, d_has_negative_cycle, sizeof(bool), cudaMemcpyDeviceToHost);
		if (!has_change || *has_negative_cycle)
			break;
		relax_swap <<<gdim, bdim>>>(relaxed_last_round, relaxed_this_round, n);
	}

	if (!*has_negative_cycle)
		cudaMemcpy(dist, d_dist, sizeof(int) * n, cudaMemcpyDeviceToHost);

	cudaFree(d_mat);
	cudaFree(d_dist);
	cudaFree(d_has_change);
	cudaFree(d_has_negative_cycle);
	cudaFree(relaxed_last_round);
	cudaFree(relaxed_this_round);
	cudaFree(relaxed_times);
	//------end of your code------
}

int main(int argc, char **argv) {
	if (argc <= 1) {
		utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
	}
	if (argc <= 3) {
		utils::abort_with_error_message("blocksPerGrid or threadsPerBlock WAS NOT FOUND!");
	}

	string filename = argv[1];
	int blockPerGrid = atoi(argv[2]);
	int threadsPerBlock = atoi(argv[3]);

	int *dist;
	bool has_negative_cycle = false;


	assert(utils::read_file(filename) == 0);
	dist = (int *)calloc(sizeof(int), utils::N);


	clock_t tb, te;
	tb = clock();
	//bellman-ford algorithm
	bellman_ford(blockPerGrid, threadsPerBlock, utils::N, utils::mat, dist, &has_negative_cycle);
	CHECK(cudaDeviceSynchronize());
	te = clock();

	std::cerr.setf(std::ios::fixed);
	std::cerr << std::setprecision(6) << "Time(s): " << ((double)(te - tb) / CLOCKS_PER_SEC) << endl;
	utils::print_result(has_negative_cycle, dist);
	free(dist);
	free(utils::mat);

	return 0;
}
