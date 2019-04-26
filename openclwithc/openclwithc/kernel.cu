#include "main.h"

/*
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
*/

__global__ void matrixKernel(long width, long height, long **result, long **Matrix, long **weight) {
	int range = MASK_RANGE / 2;
	int nowx = blockIdx.y * blockDim.y + threadIdx.y;
	int nowy = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (nowx > height - 1 || nowy > width) return;

	for (int i = -range; i < range + 1; i++) {
		for (int j = -range; j < range + 1; j++) {
			result[nowx][nowy] += (weight[i + range][j + range] * Matrix[nowx + i][nowy + j]);
		}
	}

}

// Helper function for using CUDA to add vectors in parallel.
//cudaError_t calWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//	int *dev_a = 0;
//	int *dev_b = 0;
//	int *dev_c = 0;
//	cudaError_t cudaStatus;
//
//	// Choose which GPU to run on, change this on a multi-GPU system.
//	cudaStatus = cudaSetDevice(0);
//	CUDA_CHECK_ERROR(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
//
//	// Allocate GPU buffers for three vectors (two input, one output).
//	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//	CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!\n");
//
//	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//	CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!\n");
//
//	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//	CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!\n");
//
//	// Copy input vectors from host memory to GPU buffers.
//	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//	CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy failed!\n");
//
//	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//	CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy failed!\n");
//
//	// Launch a kernel on the GPU with one thread for each element.
//	addKernel <<<1, size >>>(dev_c, dev_a, dev_b);
//
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
//	// any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//	CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy failed!\n");
//
//Error:
//	cudaFree(dev_c);
//	cudaFree(dev_a);
//	cudaFree(dev_b);
//
//	return cudaStatus;
//}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t matrixWithCuda(long **c, long **a, long b[MASK_RANGE][MASK_RANGE], unsigned int width, unsigned int height)
{
	long **dev_a;
	long **buff_a = (long **)malloc(height * sizeof(long *));
	long **dev_b;
	long **buff_b = (long **)malloc(MASK_RANGE * sizeof(long *));
	long **dev_c;
	long **buff_c = (long **)malloc(height * sizeof(long *));
	cudaError_t cudaStatus;

	memset(buff_a, NULL, sizeof(long *) * height);
	memset(buff_b, NULL, sizeof(long *) * MASK_RANGE);
	memset(buff_c, NULL, sizeof(long *) * height);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	CUDA_CHECK_ERROR(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");

	// Allocate GPU buffers for height Areea.
	LOG_MSG("Allocate GPU buffer dev c");
	cudaStatus = cudaMalloc((void***)&dev_c, sizeof(long *) * height);
	CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!\n");

	cudaStatus = cudaMemset(dev_c, NULL, height * sizeof(long *));
	CUDA_CHECK_ERROR(cudaStatus, "memory initialize Error");

	LOG_MSG("Allocate GPU buffer dev a");
	cudaStatus = cudaMalloc((void***)&dev_a, sizeof(long *) * height);
	CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!\n");

	cudaStatus = cudaMemset(dev_a, NULL, height * sizeof(long *));
	CUDA_CHECK_ERROR(cudaStatus, "memory initialize Error");


	LOG_MSG("Allocate GPU buffer dev b");
	cudaStatus = cudaMalloc((void***)&dev_b, sizeof(long *) * MASK_RANGE);
	CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!\n");

	cudaStatus = cudaMemset(dev_b, NULL, MASK_RANGE * sizeof(long *));
	CUDA_CHECK_ERROR(cudaStatus, "memory initialize Error");


	LOG_MSG("Allocate GPU buffer specific dev b and memcpy b to dev b");
	for (int rp = 0; rp < MASK_RANGE; rp++) {
		// Allocate GPU buffers for Specific Area.
		cudaStatus = cudaMalloc((void**)&(buff_b[rp]), MASK_RANGE * sizeof(long));
		CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!\n");

		cudaStatus = cudaMemset(buff_b[rp], 0, MASK_RANGE * sizeof(long));
		CUDA_CHECK_ERROR(cudaStatus, "memory initialize Error");

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(buff_b[rp], b[rp], MASK_RANGE *sizeof(long), cudaMemcpyHostToDevice);
		CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy failed!\n");
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_b, buff_b, MASK_RANGE * sizeof(long *), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy failed!\n");

	for (unsigned int rp = 0; rp < height; rp++) {
		LOG_MSG("Allocate buffer specific dev c ");

		// Allocate GPU buffers for Specific Area.
		cudaStatus = cudaMalloc((void**)&(buff_c[rp]), sizeof(long) * width);
		CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!\n");

		cudaStatus = cudaMemset(buff_c[rp], 0, width * sizeof(long));
		CUDA_CHECK_ERROR(cudaStatus, "memory initialize Error");
		
		LOG_MSG("Memory copy c to specific buffer c ");
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(buff_c[rp], c[rp], sizeof(long) * width, cudaMemcpyHostToDevice);
		CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy failed!\n");



		LOG_MSG("Allocate buffer specific dev a");
		// Allocate GPU buffers for Specific Area.
		cudaStatus = cudaMalloc((void**)&(buff_a[rp]), sizeof(long) * width);
		CUDA_CHECK_ERROR(cudaStatus, "cudaMalloc failed!\n");

		cudaStatus = cudaMemset(buff_a[rp], 0, width * sizeof(long));
		CUDA_CHECK_ERROR(cudaStatus, "memory initialize Error");

		LOG_MSG("Memory copy c to buffer c ");
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(buff_a[rp], a[rp], sizeof(long) * width, cudaMemcpyHostToDevice);
		CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy failed!\n");
	}

	LOG_MSG("Memory copy buff c to  dev c ");
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_c, buff_c, sizeof(long *) * height, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy failed!\n");

	LOG_MSG("Memory copy buff a to dev a ");
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, buff_a, sizeof(long *) * height, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy failed!\n");

	LOG_MSG("Calculating with CUDA Kernel start");
	// Launch a kernel on the GPU with one thread for each element.
	matrixKernel <<< 1, width*height >>>(width, height, dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "matrixKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching matrixKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.

	cudaStatus = cudaMemcpy(buff_c, dev_c, height * sizeof(long*), cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy failed!\n");

	for (unsigned int rp = 0; rp < height; rp++) {
		cudaStatus = cudaMemcpy(c[rp], buff_c[rp], width * sizeof(long), cudaMemcpyDeviceToHost);
		CUDA_CHECK_ERROR(cudaStatus, "cudaMemcpy failed!\n");
	}

Error:
	free(buff_a);
	free(buff_c);
	free(buff_b);
	cudaFree(dev_b);
	cudaFree(dev_a);
	cudaFree(dev_c);

	return cudaStatus;
}