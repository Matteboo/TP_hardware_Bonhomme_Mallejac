#include "Header.cuh"


int main(){
    int n = 3;
    int p = n;
    float* M1=(float*)malloc(n*p * sizeof(double));
    float* M2 = (float*)malloc(n * p * sizeof(double));
    float* Mout_cpu = (float*)malloc(n * p * sizeof(double));
    float* Mout_gpu = (float*)malloc(n * p * sizeof(double));
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    float* d_M1, * d_M2, * d_Mout;
    MatrixMult( M1, M2, Mout_cpu,n);
    cudaMalloc((void**)&d_M1, sizeof(float) * n * p);
    cudaMalloc((void**)&d_M2, sizeof(float) * n * p);
    cudaMalloc((void**)&d_Mout, sizeof(float) * n * p);
    cudaMemcpy(d_M1, M1, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    cudaMatrixMult<<<5,5>>>(d_M1, d_M2, d_Mout, n);
    cudaMemcpy(Mout_gpu, d_Mout, sizeof(float) * n * p, cudaMemcpyDeviceToHost);
    //MatrixPrint(M1, n, p);
    //MatrixPrint(M2, n, p);
    MatrixPrint(Mout_cpu,  n,  p);
    MatrixPrint(Mout_gpu, n, p);
    return 0;
}

