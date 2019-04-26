#include "main.h"

long sobel_vertical[MASK_RANGE][MASK_RANGE] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
long sobel_horizon[MASK_RANGE][MASK_RANGE] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };

int pixel_processing_gray(int Red, int Green, int Blue, int None) {
	int pixel;
	int Red_temp, Green_temp, Blue_temp;
	int Gray_temp = ((int)( (Red + Green + Blue) / 3.0));

	Red_temp = Green_temp = Blue_temp = Gray_temp;
	
	pixel = (Green_temp << 8) | (Blue_temp) | (None << 24) | (Red_temp << 16);

	return pixel;
}

cudaError_t pixel_processing_sobel(FILE *input, FILE *output, int width, int height) {
	long **pixelRGBColors, **sumResultX, **sumResultY;
	unsigned char buf[4];
	long Red_temp, Green_temp, Blue_temp, None_temp;
	long pixel;
	int ht = 0, wd = 0, totalPixel = 0;
	cudaError_t cudaStatus;

	// Allocation about pixelRGBColors
	pixelRGBColors = matrixAllocation(width, height);
	sumResultX = matrixAllocation(width, height);
	sumResultY = matrixAllocation(width, height);
	
	// Read All pixels
	while (fread(buf, 4, 1, input) != 0) {
		if (wd == width) {
			printf("%d\n", ht);
			wd = 0;
			ht++;
		}
		Red_temp = (long)buf[2];
		Green_temp = (long)buf[1];
		Blue_temp = (long)buf[0];
		None_temp = (long)buf[3];
		//printf("%ld %ld %ld %ld\n", Red_temp, Green_temp, Blue_temp, None_temp);
		pixelRGBColors[ht][wd] = pixel_processing_gray(Red_temp, Green_temp, Blue_temp, None_temp);
		wd++; totalPixel++;
	}

	printf("Real Number of Pixels : %d\n", width * height);
	printf("Total number of Pixels : %d\n", totalPixel);

	// insert here matrix sum calculation with CUDA Processing
	LOG_MSG("Processing vertical sobel");
	cudaStatus = matrixWithCuda(sumResultX, pixelRGBColors, sobel_vertical, width, height);
	CUDA_CHECK_ERROR(cudaStatus, "sumx failed\n");
	LOG_MSG("Processing horizonal sobel");
	cudaStatus = matrixWithCuda(sumResultY, pixelRGBColors, sobel_horizon, width, height);
	CUDA_CHECK_ERROR(cudaStatus, "sumy failed\n");

	// processing edge detect
	for (int i = 0; i < height - 1; i++) {
		for (int j = 0; j < width; j++) {
			// insert here matrix sum calculation with CUDA Processing
			pixel = (((sumResultX[i][j] > 0) ? sumResultX[i][j] : -sumResultX[i][j]))
					+ (((sumResultY[i][j] > 0) ? sumResultY[i][j] : -sumResultY[i][j]));
			printf("pixel value : [%d, %d] %d %d \n",i, j, sumResultX[i][j], sumResultY[i][j]);
			fwrite(&pixel, 4, 1, output);
		}
	}

Error:
	free(pixelRGBColors);
	free(sumResultX);
	free(sumResultY);

	return cudaStatus;

}

long **matrixAllocation(int width, int height) {
	long **pixelMatrix;
	// Allocation about pixelRGBColors
	pixelMatrix = (long **)malloc(sizeof(long *) *height);
	memset(pixelMatrix, NULL, sizeof(long *) *height);

	for (int i = 0; i < height; i++) {
		pixelMatrix[i] = (long *)malloc(sizeof(long) *width);
		memset(pixelMatrix[i], 0, sizeof(long) *width);
	}

	return pixelMatrix;
}