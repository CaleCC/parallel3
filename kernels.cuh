#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

__global__ void find_maximum_kernel(double *array, double *max, int *mutex, int n);
__global__ void mean_kernel(double *array, double *mean, int *mutex, int n);
__global__ void find_minimum_kernel(double *array, double *min, int *mutex, int n);
__global__ void std_kernel(double *array, double *d_std, int* mutex, int n, double mean);
__global__ void concurrent_kernel(double array,double* max,double* min, double *mean,int *mutex,int n);
#endif
