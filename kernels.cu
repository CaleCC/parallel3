#include "kernels.cuh"


__global__ void find_maximum_kernel(double *array, double *max, int *mutex, int n){
  unsigned int index = threadIdx.x + blockIdx.x*gridDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int offset = 0;

  __shared__ double cache[256];

  double temp = array[0];
  while(index + offset < n){
    temp = fmax(temp, array[index + offset]);
    offset += stride;
  }

  cache[threadIdx.x] = temp;

  __syncthreads();


  //reduction
  unsigned int i = gridDim.x/2;
  while(i != 0){
    if(threadIdx.x < i){
      cache[threadIdx.x] = fmax(cache[threadIdx.x], cache[threadIdx.x + i]);
    }

    __syncthreads();
    i /= 2;
  }

  if(threadIdx.x == 0){
    while(atomicCAS(mutex,0 ,1) != 0);
    *max = fmax(*max, cache[0]);
    atomicExch(mutex, 0);
  }
}

__global__ void find_minimum_kernel(double *array, double *min, int *mutex, int n){
  unsigned int index = threadIdx.x + blockIdx.x*gridDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int offset = 0;

  __shared__ double cache[256];

  double temp = array[0];
  while(index + offset < n){
    temp = fmin(temp, array[index + offset]);
    offset += stride;
  }

  cache[threadIdx.x] = temp;

  __syncthreads();


  //reduction
  unsigned int i = gridDim.x/2;
  while(i != 0){
    if(threadIdx.x < i){
      cache[threadIdx.x] = fmin(cache[threadIdx.x], cache[threadIdx.x + i]);
    }

    __syncthreads();
    i /= 2;
  }

  if(threadIdx.x == 0){
    while(atomicCAS(mutex,0 ,1) != 0);
    *min = fmin(*min, cache[0]);
    atomicExch(mutex, 0);
  }
}


__global__ void mean_kernel(double *array, double *mean, int *mutex, int n){
  unsigned int index = threadIdx.x + blockIdx.x*gridDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int offset = 0;

  __shared__ double cache[256];

  double temp = 0.0;
  while(index + offset < n){
    temp = temp + array[index + offset];
    offset += stride;
  }

  cache[threadIdx.x] = temp;

  __syncthreads();


  //reduction
  // unsigned int i = gridDim.x/2;
  // while(i != 0){
  //   if(threadIdx.x < i){
  //     cache[threadIdx.x] = fmin(cache[threadIdx.x], cache[threadIdx.x + i]);
  //   }
  //
  //   __syncthreads();
  //   i /= 2;
  // }
  unsigned int i = gridDim.x / 2;
  while(i != 0){
    if(threadIdx.x < i){
      cache[threadIdx.x] += cache[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }
  __syncthreads();
  if(threadIdx.x == 1){
    while(atomicCAS(mutex, 0 ,1) != 0);
    *mean += cache[0];
    atomicExch(mutex, 0);
  }
}



__global__ void std_kernel(double *array, double* d_std, int *mutex, int n, double mean){
  unsigned int index = threadIdx.x + blockIdx.x*gridDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int offset = 0;

  __shared__ double cache[256];

  double temp = 0.0;
  while(index + offset < n){
    temp = temp + (array[index + offset]-mean)*(array[index + offset]-mean);
    offset += stride;
  }

  cache[threadIdx.x] = temp;

  __syncthreads();


  //reduction
  // }
  unsigned int i = gridDim.x / 2;
  while(i != 0){
    if(threadIdx.x < i){
      cache[threadIdx.x] += cache[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }
  __syncthreads();
  if(threadIdx.x == 1){
    while(atomicCAS(mutex, 0 ,1) != 0);
    *d_std += cache[0];
    atomicExch(mutex, 0);
  }


}

__global__ void concurrent_kernel(double array,double* max,double* min, double *mean, int *mutex, int n){
  unsigned int index = threadIdx.x + blockIdx.x*gridDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int offset = 0;

  __shared__ double cache_max[256];
  __shared__ double cache_min[256];
  __shared__ double cache_mean[256];

  double temp_max = array[0];
  double temp_min = array[0];
  double temo_mean = 0;
  while(index + offset < n){
    temp_max = fmax(temp_max, array[index + offset]);
    temp_min = fmin(temp_min,array[index + offset]);
    temp_mean += array[index + offset];
    offset += stride;
  }

  cache_max[threadIdx.x] = temp_max;
  cahce_min[threadIdx.x] = temp_min;
  cache_mean[threadIdx.x] = temp_mean;

  __syncthreads();


  //reduction
  unsigned int i = gridDim.x/2;
  while(i != 0){
    if(threadIdx.x < i){
      cache_max[threadIdx.x] = fmax(cache_max[threadIdx.x], cache_max[threadIdx.x + i]);
      cache_min[threadIdx.x] = fmin(cache_min[threadIdx.x], cache_min[threadIdx.x + i]);
      cache_mean[threadIdx.x] += cache_mean[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }


  if(threadIdx.x == 0){

    while(atomicCAS(mutex,0 ,1) != 0);
    *max = fmax(*max, cache_max[0]);
    *min = fmin(*min, cache_min[0]);
    *mean += cache_mean[0];
    atomicExch(mutex, 0);
  }
}
