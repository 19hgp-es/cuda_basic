#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <memory.h>

#define MASK_RANGE 3
#define MAX_HEIGHT 1080
#define THREAD_BLOCKS 16
#define CUDA_CHECK_ERROR(err, msg) \
	if(err != cudaSuccess) { \
		fprintf(stderr, msg); \
		goto Error; \
	}

#define LOG_MSG(msg) \
	SetConsoleTextAttribute(hOUT, FOREGROUND_GREEN); \
	printf("[LOG] %s\n", msg); \
	SetConsoleTextAttribute(hOUT, FOREGROUND_INTENSITY);

extern HANDLE hOUT;

extern long sobel_vertical[MASK_RANGE][MASK_RANGE];
extern long sobel_horizon[MASK_RANGE][MASK_RANGE];


//cudaError_t calWithCuda(int *c, const int *a, const int *b, unsigned int size);

cudaError_t checkMyDeviceStatus(void);

//__global__ void addKernel(int *c, const int *a, const int *b);

int pixel_processing_gray(int Red, int Green, int Blue, int None);

cudaError_t pixel_processing_sobel(FILE *input, FILE *output, int width, int height);

long **matrixAllocation(int width, int height);

cudaError_t matrixWithCuda(long **c, long **a, long b[MASK_RANGE][MASK_RANGE], unsigned int width, unsigned int height);

__global__ void matrixKernel(long width, long height, long **result, long **Matrix, long **weight);

