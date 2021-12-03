#include "Header.cuh"

void MatrixInit(float* M, int n, int p) {
	
	for (int i =0; i < n; i += 1) {
		for (int j = 0; j < n; j += 1) {
			int seed = n * j + i;

			M[i + n * j] = (rand() / (double)(RAND_MAX)) * 2 - 1;
		
		}
	}
}

void MatrixPrint(float* M, int n, int p) {
	int i = 0;
	int j = 0;
	for (i; i < n; i++) {
		for (j = 0; j < p; j++) {
			printf("  %f  ", M[i + n * j]);
		}
		printf("\n");
	}
}

void MatrixAdd(float* M1, float* M2, float* Mout, int n, int p) {
	int i = 0;
	int j = 0;
	for (i; i < n; i++) {
		for (j = 0; j < p; j++) {
			Mout[i + n * j] = M1[i + n * j] + M2[i + n * j];
		}
	}

}



__global__ void cudaMatrixAdd(float* M1, float* M2, float* Mout, int n, int p) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n * p) {
		Mout[i] = M1[i] + M2[i];

	}
}


void MatrixMult(float* M1, float* M2, float* Mout, int n) {
	for (int i = 0; i < n; ++i){
		for (int j = 0; j < n; ++j){
			float tmp = 0.0;
			for (int h = 0; h < n; ++h){
				tmp += M1[i * n + h] * M2[h * n + j];
			}
			Mout[i * n + j] = tmp;
		}
	}
}


__global__ void cudaMatrixMult(float* M1, float* M2, float* Mout, int n) {
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int i = k / n;		//ligne
	int j = k - n * i;	//colonne
	if (i < n * n) {
		float tmp = 0.0;
		for (int h = 0; h < n; ++h) {
			tmp += M1[i * n + h] * M2[h * n + j];
		}
		Mout[i * n + j] = tmp;
	}
}


