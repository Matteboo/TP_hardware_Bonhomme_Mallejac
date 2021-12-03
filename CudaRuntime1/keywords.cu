#include "Header.cuh"
 
// __device__ keyword specifies a function that is run on the device and called from a kernel (1a)
void GPUFunction(){ 
    printf("\tHello, from the GPU! (1a)\n");
}
 
// This is a kernel that calls a decive function (1b)
void kernelA(){
    GPUFunction();
}
 
// __host__ __device__ keywords can be specified if the function needs to be 
//                     available to both the host and device (2a)
void versatileFunction(){
    printf("\tHello, from the GPU or CPU! (2a)\n");
}
 
// This is a kernel that calls a function on the device (2b)
void kernelB(){
    versatileFunction();
}
 