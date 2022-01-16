#include "Header.cuh"


int main(){
    int n = 4;
    float* raw_data=(float*)malloc(n*n * sizeof(double));
    MatrixInit(raw_data, n, n);
    int p = 2;
    int depth = 2;
    float* C1_data = (float*)malloc(p* p * depth * sizeof(double));
    MatrixInit3D_value(C1_data, p, p, depth, 0.5);
    int m = 1;
    float* S1_data = (float*)malloc(m * m * depth* sizeof(double));
    MatrixInit3D_value(S1_data, m, m, depth, 0);
    int kernel_size = 3;
    float* C1_kernel = (float*)malloc(kernel_size * kernel_size * depth * sizeof(double));
    MatrixInit3D(C1_kernel, kernel_size, kernel_size, depth);
    // creation sur le GPU
    float *d_raw_data, *d_C1_data,*d_S1_data, *d_C1_kernel;
    cudaMalloc((void**)&d_raw_data,n *n * sizeof(double));
    cudaMalloc((void**)&d_C1_data, p * p * depth * sizeof(double));
    cudaMalloc((void**)&d_S1_data, m * m * depth * sizeof(double));
    cudaMalloc((void**)&d_C1_kernel, kernel_size * kernel_size * depth * sizeof(double));

    //envoie sur le GPU
    cudaMemcpy(d_raw_data, raw_data, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, kernel_size * kernel_size * depth * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_C1_data, C1_data, p * p * depth * sizeof(double), cudaMemcpyHostToDevice);

    //opérations
    Conv2D<<<5,5>>>(d_raw_data, d_C1_data, d_C1_kernel, p, kernel_size,depth);
    activation_tanh << <5, 5 >> > (d_C1_data, p * p * depth * sizeof(double));
    subsampling2D <<<5, 5 >>> (d_C1_data, d_S1_data, m, 2, depth);


    //renvoie sur le CPU
    cudaMemcpy(C1_data, d_C1_data, p * p * depth * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, m * m * depth * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    MatrixPrint3D(raw_data, n, n, 1);
    MatrixPrint3D(C1_kernel, kernel_size, kernel_size, depth);
    MatrixPrint3D(C1_data, p, p, depth);
    //MatrixPrint3D(S1_data, m, m, depth);
    return 0;
}

