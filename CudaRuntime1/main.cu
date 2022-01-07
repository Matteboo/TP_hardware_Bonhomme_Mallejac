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
    //clock_t begin_cpu = clock();
    MatrixMult( M1, M2, Mout_cpu,n);
    //clock_t end_cpu = clock();
    
    cudaMalloc((void**)&d_M1, sizeof(float) * n * p);
    cudaMalloc((void**)&d_M2, sizeof(float) * n * p);
    cudaMalloc((void**)&d_Mout, sizeof(float) * n * p);
    cudaMemcpy(d_M1, M1, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    //clock_t begin_gpu = clock();
    cudaMatrixMult<<<5,5>>>(d_M1, d_M2, d_Mout, n);
    //clock_t end_gpu = clock();
    cudaMemcpy(Mout_gpu, d_Mout, sizeof(float) * n * p, cudaMemcpyDeviceToHost);
    //double time_spent_cpu = (double) (end_cpu-begin_cpu) / CLOCKS_PER_SEC;
    //double time_spent_gpu = (double) (end_gpu-begin_gpu) / CLOCKS_PER_SEC;
    //MatrixPrint(M1, n, p);
    //MatrixPrint(M2, n, p);
    MatrixPrint(Mout_cpu,  n,  p);
    MatrixPrint(Mout_gpu, n, p);
    //printf("Execution time cpu = %f\n", time_spent_cpu);
    //printf("Execution time gpu = %f\n", time_spent_gpu);
    return 0;
}

