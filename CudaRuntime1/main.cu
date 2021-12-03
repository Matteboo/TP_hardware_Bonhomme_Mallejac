#include "Header.cuh"


int main(){
    int Nx = 3;
    int Ny = 3;
    int thread = 32;
    int block_x = ceil(Nx + thread - 1) / thread;
    int block_y = ceil(Ny + thread - 1) / thread;
    dim3 THREADS(thread, thread);
    dim3 BLOCKS(block_y, block_x);
    double* M;
    //int N_thread = 1024;
    //int N_block = (N + N_thread) / N_thread;
    M = (double*)malloc(Nx*Ny);
    curandState* dev_random;
    cudaMalloc((void**)&dev_random, Nx * Ny* thread* thread * sizeof(curandState));
    MatrixInit<<<BLOCKS, THREADS >>>(M, Nx, Ny, dev_random);

    MatrixPrint( M,  Nx, Ny);
    return 0;
}

