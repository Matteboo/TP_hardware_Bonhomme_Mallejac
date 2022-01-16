#include "Header.cuh"

void MatrixInit(float* M, int n, int p) {
	
	for (int i =0; i < n; i += 1) {
		for (int j = 0; j < p; j += 1) {
			

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
	// fill the matrix with the value v
	for (int i = 0; i < n; i += 1) {
		for (int j = 0; j < p; j += 1) {
			for (int k = 0; k < d; k += 1) {
				M[i + n * j + k * n * p] = v;
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

	int f = k / (size_M_out * size_M_out);						//face de la sortie
	int i = (k - f * size_M_out * size_M_out) / size_M_out;		//ligne de la sortie
	int j = k - f * size_M_out * size_M_out - size_M_out * i;	//colonne de la sortie
	//k=f*siz*size+i*size+j
	if (k < size_M_out * size_M_out*depth) {
		//printf("k=  %u     f=%u       i= %u      j= %u \n", k, f, i, j);
		int bound = (size_kernel - 1) / 2;
		float tmp = 0.0;
		for (int h = 0; h < size_kernel; h++) {//indice pour parcourir les lignes
			for (int l = 0; l < size_kernel; l++) {// indice pour parcourir les colonnes

				tmp += kernel[f* size_kernel * size_kernel +h * size_kernel + l] * M_in[(i+h)*(size_M_out+ 2*bound) + j+l];
			}
		}
		M_out[f* size_M_out* size_M_out+i * size_M_out + j] = tmp;
	}
}

__global__ void subsampling2D(float* M_in, float* M_out, int size_M_out, int size_window, int depth) {
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	int f = k / (size_M_out * size_M_out);						//face de la sortie
	int i = (k - f * size_M_out * size_M_out) / size_M_out;		//ligne de la sortie
	int j = k - f * size_M_out * size_M_out - size_M_out * i;	//colonne de la sortie
	//k=f*siz*size+i*size+j
	//printf("k=  %u     f=%u       i= %u      j= %u \n", k,f,i,j);

	if (k < size_M_out * size_M_out * depth) {
		//printf("k=  %u     f=%u       i= %u     j= %u \n", k, f, i, j);
		double tmp = 0.0; // variable pour stocker la valeur finale
		for (int h = 0; h < size_window; ++h) {
			for (int l = 0; l < size_window; l++) {
				tmp +=M_in[f * size_M_out * size_M_out*size_window * size_window + (i+h)* size_M_out* size_window+j+l]/(size_window* size_window);
				//printf("index= %u  M[]=%f\n ", f * size_M_out * size_M_out * size_window * size_window + (i + h) * size_M_out * size_window + j + l, M_in[f * size_M_out * size_M_out * size_window * size_window + (i + h) * size_M_out * size_window + j + l]);
			}
		}
		M_out[f * size_M_out * size_M_out + i * size_M_out + j] = tmp;
		

	}
	
}

void MatrixPrint3D(float* M, int n, int p, int q) {
	int i = 0;
	int j = 0;
	int k = 0;
	for (k = 0; k < q; k++) {
		for (j = 0; j < p; j++) {
			for (i = 0; i < n; i++)			 {
				printf("  %f  ", M[i + n * j+k*n*p]);
			}
			printf("\n");
		}
		printf("\n");//double saut de ligne quand on change de face
		printf("\n");
	}
}

__global__ void activation_tanh(float* M, int size) {
	//applique tanh à tout les élément de la matrice M
	//Size est le nombre d'ELEMENTS de la matrice M (ligne*colonne*profondeur)
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k < size) {
		M[k] = tanh(M[k]);
	}

}