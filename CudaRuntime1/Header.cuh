
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <curand.h>

#define N 10000000

#define MAX_ERR 1e-6
void GPUFunction();
void kernelA();
void versatileFunction();
void kernelB();
__global__ void vector_add(float* out, float* a, float* b, int n);
__global__ void MatrixInit(double* M, int n, int p, curandState* states);
void MatrixPrint(double* M, int Nx, int Ny);
