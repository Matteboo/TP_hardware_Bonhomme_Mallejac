#include "Header.cuh"

void MatrixInit(float* M, int n, int p) {
	
	for (int i =0; i < n; i += 1) {
		for (int j = 0; j < p; j += 1) {
			int seed = n * j + i;

			M[i + n * j] = (rand() / (double)(RAND_MAX)) * 2 - 1;
		
		}
	}
}


void MatrixInit3D(float* M, int n, int p, int d) {

	for (int i = 0; i < n; i += 1) {
		for (int j = 0; j < p; j += 1) {
			for (int l = 0; l < d; l += 1) {
				M[i + n * j + l * n * p] = (rand() / (double)(RAND_MAX)) * 2 - 1;
			}
		}
	}
}

void MatrixInit3D_value(float* M, int n, int p, int d , float v) {

	for (int i = 0; i < n; i += 1) {
		for (int j = 0; j < p; j += 1) {
			for (int l = 0; l < d; l += 1) {
				M[i + n * j + l * n * p] = v;
			}
		}
	}
}


void MatrixInit_Value(float* M, int n, int p, float v) {

	for (int i = 0; i < n; i += 1) {
		for (int j = 0; j < p; j += 1) {
			M[i + n * j] = v;
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
	if (k< n * n) {
		float tmp = 0.0;
		for (int h = 0; h < n; ++h) {
			tmp += M1[i * n + h] * M2[h * n + j];
		}
		Mout[i * n + j] = tmp;
	}
}

__global__ void Conv2D(float* M_in, float* M_out, float* kernel,int size_M_out, int size_kernel, int depth) {
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int f = k / depth;										//face de la sortie
	int i = (k-f* size_M_out* size_M_out) / size_M_out;		//ligne de la sortie
	int j = k - size_M_out * i;								//colonne de la sortie
	if (k < size_M_out * size_M_out*depth) {
		int bound = (size_kernel - 1) / 2;
		float tmp = 0.0;
		for (int h = -bound; h < bound +1; ++h) {
			for (int l = -bound; l < bound + 1; l++) {
				tmp += kernel[f* size_kernel * size_kernel +(h+bound) * size_kernel + (l+bound)] * M_in[(i+h) * (size_M_out+ bound) + j+l];
			}
		}
		M_out[f* size_M_out* size_M_out+i * size_M_out + j] = tmp;

	}
}

__global__ void subsampling2D(float* M_in, float* M_out, int size_M_in, int size_window, int depth) {
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int f = k / depth;										//face de la sortie
	int i = (k - f * size_M_out * size_M_out) / size_M_out;		//ligne de la sortie
	int j = k - size_M_out * i;								//colonne de la sortie
	if (k < size_M_out * size_M_out * depth) {
		float tmp = 0.0;
		for (int h = 0; h < size_window; ++h) {
			for (int l = 0; l < size_window; l++) {
				tmp += 1/(size_window* size_window) * M_in[f * size_M_out * size_M_out*size_window * size_window + (i+h)* size_M_out* size_window+j+l];
			}
		}
		M_out[f * size_M_out * size_M_out + i * size_M_out + j] = tmp;

	}
}

void MatrixPrint3D(float* M, int n, int p, int q) {
	int i = 0;
	int j = 0;
	int k = 0;
	for (i; i < n; i++) {
		for (j = 0; j < p; j++) {
			for (k = 0; k < q; k++) {
				printf("  %f  ", M[i + n * j+k*n*p]);
			}
			printf("\n");
		}
		printf("\n");
		printf("\n");
	}
}
