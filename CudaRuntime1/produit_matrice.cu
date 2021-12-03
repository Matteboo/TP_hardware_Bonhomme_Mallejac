#include "Header.cuh"

__global__ void MatrixInit(double* M, int n, int p, curandState* states) {
	
	for (int i = (threadIdx.x + blockIdx.x * blockDim.x); i < n; i += (blockDim.x * gridDim.x)) {
		for (int j = (threadIdx.y + blockIdx.y * blockDim.y); j < p; j += (blockDim.y * gridDim.y)) {
			int seed = n * j + i;

			curand_init(seed, i, 0, &states[n * j + i]);
			M[i + n * j] = 1;
				//curand_uniform(&states[n * j + i]);
		
		}
	}
}

void MatrixPrint(double* M, int Nx, int Ny) {
	printf("coucou \n");
	int i = 0;
	int j = 0;
	for (i; i < Nx; i++) {
		for (j = 0; j < Ny; j++) {
			printf("  %f  ", M[i + Nx * j]);
		}
		printf("\n");
	}
}