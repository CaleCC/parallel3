/*written by Cheng Chen
parallel computing second project part 1 12/4/2016*/
#include <iostream>
#include <stdlib.h>

#include <algorithm>
#include <ctime>
#include "kernels.cuh"
using namespace std;


int randNum() {
	double ran = (double)rand();
	return ran;
}

// double timeTransfer(struct timeval start, struct timeval end){
//   double startInuSec = start.tv_sec * 1000 * 1000 + start.tv_usec;
//   double endInuSec = end.tv_sec * 1000 * 1000 + end.tv_usec;
//   return (endInuSec - startInuSec)/1000000.0;
// }

void concurrent_all(double array, int n){
	double *h_max;
	double *h_std;
	double *h_mean;
	double *h_min;

	h_max = (double*)malloc(sizeof(double));
	h_std = (double*)malloc(sizeof(double));
	h_mean = (double*)malloc(sizeof(double));
	h_min = (double*)malloc(sizeof(double));

	double *d_max;
	double *d_min;
	double *d_mean;
	double *d_std;
	double *d_array;
	int *d_mutex;
	//cudaMalloc((void**)&d_array, n*sizeof(double));
	cudaMalloc((void**)&d_max, sizeof(double));
	cudaMalloc((void**)&d_min, sizeof(double));
	cudaMalloc((void**)&d_mean, sizeof(double));
	cudaMalloc((void**)&d_std, sizeof(double));
	cudaMalloc((void**)&d_array, n*sizeof(double));

	cudaMemset(d_max, 0, sizeof(double));
	cudaMemset(d_min, 0, sizeof(double));
	cudaMemset(d_std, 0, sizeof(double));
	cudaMemset(d_mean, 0, sizeof(double));
	cudaMemset(d_mutex, 0, sizeof(int));

	cudaMemcpy(d_array, h_array, N*size(double), cudaMemcpyHostToDevice);

	double gpu_elapsed_time;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);

	dim3 gridSize = 256;
	dim3 blockSize = 256;
	cudaEventRecord(gpu_start, 0);
	concurrent_kernel<<<gridSize, blockSize>>>(d_array, d_max, d_min, d_mean, d_mutex, n);
	cudaMemcpy(h_mean, d_mean, sizeof(double), cudaMemcpyDeviceToHost);
	*h_mean = *h_mean / n;
	std_kernel<<<gridSize, blockSize>>>(d_array, d_std, d_mutex, n, *h_mean);
	cudaMemcpy(h_std, d_std, sizeof(double), cudaMemcpyDeviceToHost);
	*d_std = sqrt(*d_std/n);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);
	cudaMemcpy(h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_min, d_min, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_std, d_std, sizeof(double), cudaMemcpyDeviceToHost);

	std::cout<<"max "<<h_max<<" min "<<h_min<<" mean "<<h_mean << " Stand d "<< h_std<<endl;
	std::cout<<"array size "<< n<<" time "<<gpu_elapsed_time<<endl;

	//cpu version
	cout<<"-----------cpu Version concurrent---------------------"<<endl;
	double max=array[0], min=array[0], mean = 0, standDeviation = 0;
	for(int i = 0; i < n; ++i){
		if(max < array[i]){
			max = array[i];
		}
		if(min < array[i]){
			min = array[i];
		}
		mean += array[i];
	}
	mean = mean / n;
	clock_t cpu_start = clock();
	for(int i = 0; i < n; i++){
		standDeviation+=(array[i] - mean)*(array[i] - mean);
	}
	standDeviation =sqrt( standDeviation/n);
	clock_t cpu_stop = clock();
	double cpu_elapsed_time = 1000*(cpu_stop - cpu_start)/CLOCKS_PER_SEC;

	std::cout<<"max "<<max<<" min "<<min<<" mean "<<mean << " Stand d "<<standDeviation<<endl;
	std::cout<<"array size "<< n<<" time "<<cpu_elapsed_time<<endl;

}

int main(int argc, char *argv[]) {
	//set the total number
	int totalNum = 50 * 1000 * 1000;
	double *d_array;
	double *d_max;
	int *d_mutex;
	double *h_max;


	//allocate memory space for the random numbers
	double* nums = (double*)malloc(totalNum * sizeof(double));
	h_max = (double*)malloc(sizeof(double));
	cudaMalloc((void**)&d_array, totalNum * sizeof(double));
	cudaMalloc((void**)&d_max, sizeof(double));
	cudaMalloc((void**)&d_mutex, sizeof(int));
	cudaMemset(d_max, 0, sizeof(float));
	cudaMemset(d_mutex, 0, sizeof(int));

	//fill array with data
	srand(time(NULL));
	for (int i = 0; i < totalNum; ++i) {
		nums[i] = randNum();
	}

	//set up timing variables
	float gpu_elapased_time;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);

	//copy from host to device
	cudaMemcpy(d_array, nums, totalNum * sizeof(double), cudaMemcpyHostToDevice);
	dim3 gridSize = 256;
	dim3 blockSize = 256;


	//call kernel
	printf("---------GPU version find max-------------------\n");

	cudaEventRecord(gpu_start, 0);
	find_maximum_kernel <<<gridSize, blockSize>>>(d_array, d_max, d_mutex, totalNum);


	//copy from GPU to host
	cudaMemcpy(h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapased_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

	std::cout << "Maximum number fouind on gpu was: " << *h_max << std::endl;
	std::cout << "size of arrays " << totalNum << " time " << gpu_elapased_time << endl;




	//cpu version
	printf("------------CPU version find the maximum---------------\n");
	clock_t cpu_start = clock();
	double maxNum = nums[0];
	for (int i = 0; i < totalNum; i++) {
		if (maxNum < nums[i]) {
			maxNum = nums[i];
		}
	}
	clock_t cpu_stop = clock();
	clock_t cpu_elapsed_time = 1000*(cpu_stop - cpu_start)/CLOCKS_PER_SEC;
	cout << "size of arrays " << totalNum << " time " << cpu_elapsed_time << endl;
	cout << "the max number is " << maxNum << endl;



	printf("---------GPU version find min-------------------\n");
	//allocate memory space for the random numbers
	double* h_min;
	h_min = (double*)malloc(sizeof(double));
	double* d_min;
	cudaMalloc((void**)&d_min, sizeof(double));
	cudaMemset(d_min, 0, sizeof(float));
	cudaMemset(d_mutex, 0, sizeof(int));



	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);



	cudaEventRecord(gpu_start, 0);
	find_minimum_kernel <<<gridSize, blockSize>>>(d_array, d_min, d_mutex, totalNum);


	//copy from GPU to host
	cudaMemcpy(h_min, d_min, sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapased_time, gpu_start, gpu_stop);

	std::cout << "Minimun number found on gpu was: " << *h_min << std::endl;
	std::cout << "size of arrays " << totalNum << " time " << gpu_elapased_time << endl;




	//cpu version
	printf("------------CPU version find the minimun---------------\n");
	//gettimeofday(&start, NULL);
	double minimum = nums[0];
	cpu_start = clock();
	for (int i = 0; i < totalNum; i++) {
		if (maxNum < nums[i]) {
			maxNum = nums[i];
		}
	}
	cpu_stop = clock();
	cpu_elapsed_time = 1000*(cpu_stop - cpu_start)/CLOCKS_PER_SEC;
	cout << "size of arrays " << totalNum << " time " << cpu_elapsed_time << endl;
	cout << "the max number is " << minimum << endl;


		printf("---------GPU version arithmetic mean-------------------\n");
		//allocate memory space for the random numbers
		double* h_mean;
		h_mean = (double*)malloc(sizeof(double));
		double* d_mean;
		cudaMalloc((void**)&d_mean, sizeof(double));
		cudaMemset(d_mean, 0, sizeof(float));
		cudaMemset(d_mutex, 0, sizeof(int));


		//set up timing variables
		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);



		cudaEventRecord(gpu_start, 0);
		mean_kernel <<<gridSize, blockSize>>> (d_array, d_mean, d_mutex, totalNum);


		//copy from GPU to host
		cudaMemcpy(h_mean, d_mean, sizeof(double), cudaMemcpyDeviceToHost);
		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapased_time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		*h_mean = *h_mean/totalNum;
		std::cout << "mean of the array calculate by gpu was: " << *h_mean<< std::endl;
		std::cout << "size of arrays " << totalNum << " time " << gpu_elapased_time << endl;




		//cpu version
		printf("------------CPU version find the arithmetic mean---------------\n");
		//gettimeofday(&start, NULL);
		double mean = nums[0];
		cpu_start = clock();
		for (int i = 1; i < totalNum; i++) {
				mean += nums[i];
		}
		mean = mean/totalNum;
		cpu_stop = clock();
		cpu_elapsed_time = 1000*(cpu_stop - cpu_start)/CLOCKS_PER_SEC;
		cout << "size of arrays " << totalNum << " time " << cpu_elapsed_time << endl;
		cout << "the  mean is " << mean << endl;

		//GPU version std
		printf("-----------------GPU version STD--------------------\n");
		double *h_std = (double*)malloc(sizeof(double));
		double *d_std;
		cudaMalloc((void**)&d_std, size(double));
		cudaMemset(d_std, 0, sizepf(double));
		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);

		cudaEventRecord(gpu_start, 0);
		std_kernel<<<gridSize, blockSize>>>(d_array, d_std, d_mutex, totalNum, *h_mean);
		cudaEventRecord(gpu_stop,0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapased_time, gpu_start, gpu_stop);
		cudaMemcpy(h_std, d_std, sizeof(double), cudaMemcpyDeviceToHost);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		*h_std = sqrt(*h_std/totalNum);
		std::cout<<"GPU std is " << *h_std<<endl;
		std::cout<<"the gpu took: "<<gpu_elapased_time<<endl;

		//run the cpu version std
		printf("------------CPU version find the STD---------------\n");
		//gettimeofday(&start, NULL);
		double stand_d= 0;
		cpu_start = clock();
		for (int i = 0; i < totalNum; i++) {
				stand_d += (nums[i]-mean)*(nums[i]-mean);
		}
		stand_d = stand_d/totalNum;
		stand_d = sqrt(stand_d);
		cpu_stop = clock();
		cpu_elapsed_time = 1000*(cpu_stop - cpu_start)/CLOCKS_PER_SEC;
		cout << "size of arrays " << totalNum << " time " << cpu_elapsed_time << endl;
		cout << "the  stand deviation is " << stand_d << endl;

		//free the numbers
		free(h_max);
		free(h_min);
		free(h_std);
		free(h_mean);
		//free gpu

	cudaFree(d_max);
	cudaFree(d_min);
	cudaFree(d_std);
	cudaFree(d_mean);

	concurrent_all(h_array, totalNum);

}
