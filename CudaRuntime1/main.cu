#include "Header.cuh"


int main(){
    int n = 6;
    float* raw_data=(float*)malloc(n*n * sizeof(double));
    MatrixInit(raw_data, 32, 32);
    int p = 4;
    int depth = 2;
    float* C1_data = (float*)malloc(p* p * depth * sizeof(double));
    MatrixInit3D_value(C1_data, p, p, depth, 0);
    int m = 1;
    float* S1_data = (float*)malloc(m * m * depth* sizeof(double));
    MatrixInit3D_value(C1_data, m, m, depth, 0);
    int kernel_size = 3;
    float* C1_kernel = (float*)malloc(kernel_size * kernel_size * depth * sizeof(double));
    MatrixInit3D(C1_data, kernel_size, kernel_size, depth);

    Conv2D<<<5,5>>>(raw_data, C1_data, C1_kernel, p, kernel_size,depth);
    return 0;
}

