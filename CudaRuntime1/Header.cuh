
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
void MatrixInit(float* M, int n, int p);
void MatrixPrint(float* M, int Nx, int Ny);
void MatrixAdd(float* M1, float* M2, float* Mout, int n, int p);
__global__ void cudaMatrixAdd(float* M1, float* M2, float* Mout, int n, int p);
void MatrixMult(float* M1, float* M2, float* Mout, int n);
__global__ void cudaMatrixMult(float* M1, float* M2, float* Mout, int n);
void MatrixInit3D_value(float* M, int n, int p, int d, float v);
void MatrixInit3D(float* M, int n, int p, int d);
__global__ void Conv2D(float* M_in, float* M_out, float* kernel, int size_M_out, int size_kernel, int depth);