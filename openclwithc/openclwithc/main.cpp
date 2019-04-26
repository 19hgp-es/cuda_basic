#include "main.h"

HANDLE hOUT = GetStdHandle(STD_OUTPUT_HANDLE);

int main(void)
{
	FILE *bmpfpr, *bmpfpw;

	char file[20] = "";
	unsigned char info[54];

	int num = 0;

	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };
	cudaError_t cudaStatus;

	printf("This Program is CUDA with image processing.\n");
	
	// check My device status
	cudaStatus = checkMyDeviceStatus();
	CUDA_CHECK_ERROR(cudaStatus, "checkMydevice failed!\n")
	
	printf("Image name (ex : lena.bmp) : ");
	scanf_s("%s", file, (unsigned int)sizeof(file));

	if (fopen_s(&bmpfpr, file, "rb") != 0) {
		printf("File not found\n");
		goto Error;
	}

	if (fopen_s(&bmpfpw, "output.bmp", "wb") != 0) {
		printf("File not found\n");
		goto Error;
	}

	fread(info, sizeof(unsigned char), 54, bmpfpr);
	int width = *(int *)&info[18];
	int height = *(int *)&info[22];
	printf("------- picture info ---------\n");
	printf("--> height : %d\n", height);
	printf("--> width : %d\n", width);
	printf("------------------------------\n");

	LOG_MSG("Write header in Processed output file");
	fwrite(info, sizeof(unsigned char), 54, bmpfpw);

	LOG_MSG("Processing start");
	cudaStatus = pixel_processing_sobel(bmpfpr, bmpfpw, width, height);
	CUDA_CHECK_ERROR(cudaStatus, "processing Error\n");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	CUDA_CHECK_ERROR(cudaStatus, "cudaDeviceReset failed!\n");

Error:

	fclose(bmpfpr);
	fclose(bmpfpw);
	system("pause");
	return 0;
}