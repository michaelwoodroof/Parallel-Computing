#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cuda.h>
#define BW 16 // Block Width
#define BH 32 // Block Height
#define COUNT 0


// Kernel Function handles first nested for loop
__global__ void kernelBlur(int *d_Rnew, int *d_Gnew, int *d_Bnew, int *d_R, int *d_G, int *d_B, int rowsize, int colsize) {
	// Set-up
	int row = blockIdx.y*blockDim.y + threadIdx.y;
    	int col = blockIdx.x*blockDim.x + threadIdx.x;
	// Run Some Calculations
	if (col < colsize && row < rowsize) {
		if (row != 0 && row != (rowsize-1) && col != 0 && col != (colsize-1)) {
			d_Rnew[row * colsize + col] = (d_R[(row + 1) * colsize + col]+d_R[(row - 1) * colsize + col]+d_R[row * colsize + (col + 1)]+d_R[row * colsize + (col - 1)])/4;
			d_Gnew[row * colsize + col] = (d_G[(row + 1) * colsize + col]+d_G[(row - 1) * colsize + col]+d_G[row * colsize + (col + 1)]+d_G[row * colsize + (col - 1)])/4;
			d_Bnew[row * colsize + col] = (d_B[(row + 1) * colsize + col]+d_B[(row - 1) * colsize + col]+d_B[row * colsize + (col + 1)]+d_B[row * colsize + (col - 1)])/4;
		}
		else if (row == 0 && col != 0 && col != (colsize-1)){
			d_Rnew[row * colsize + col] = (d_R[(row + 1)  * colsize + col]+d_R[row * colsize + (col + 1)]+d_R[row * colsize + (col - 1)])/3;
			d_Gnew[row * colsize + col] = (d_G[(row + 1)  * colsize + col]+d_G[row * colsize + (col + 1)]+d_G[row * colsize + (col - 1)])/3;
			d_Bnew[row * colsize + col] = (d_B[(row + 1)  * colsize + col]+d_B[row * colsize + (col + 1)]+d_B[row * colsize + (col - 1)])/3;
		}
		else if (row == (rowsize-1) && col != 0 && col != (colsize-1)){
			d_Rnew[row * colsize + col] = (d_R[(row - 1) * colsize + col]+d_R[row * colsize + (col + 1)]+d_R[row * colsize + (col - 1)])/3;
			d_Gnew[row * colsize + col] = (d_G[(row - 1) * colsize + col]+d_G[row * colsize + (col + 1)]+d_G[row * colsize + (col - 1)])/3;
			d_Bnew[row * colsize + col] = (d_B[(row - 1) * colsize + col]+d_B[row * colsize + (col + 1)]+d_B[row * colsize + (col - 1)])/3;
		}
		else if (col == 0 && row != 0 && row != (rowsize-1)){
			d_Rnew[row * colsize + col] = (d_R[(row + 1) * colsize + col]+d_R[(row - 1) * colsize + col]+d_R[row * colsize + (col + 1)])/3;
			d_Gnew[row * colsize + col] = (d_G[(row + 1) * colsize + col]+d_G[(row - 1) * colsize + col]+d_G[row * colsize + (col + 1)])/3;
			d_Bnew[row * colsize + col] = (d_B[(row + 1) * colsize + col]+d_B[(row - 1) * colsize + col]+d_B[row * colsize + (col + 1)])/3;
		}
		else if (col == (colsize-1) && row != 0 && row != (rowsize-1)){
			d_Rnew[row * colsize + col] = (d_R[(row + 1) * colsize + col]+d_R[(row - 1) * colsize + col]+d_R[row * colsize + (col + 1)])/3;
			d_Gnew[row * colsize + col] = (d_G[(row + 1) * colsize + col]+d_G[(row - 1) * colsize + col]+d_G[row * colsize + (col + 1)])/3;
			d_Bnew[row * colsize + col] = (d_B[(row + 1) * colsize + col]+d_B[(row - 1) * colsize + col]+d_B[row * colsize + (col + 1)])/3;
		}
		else if (row==0 &&col==0){
			d_Rnew[row * colsize + col] = (d_R[row * colsize + (col + 1)]+d_R[(row + 1) * colsize + col])/2;
			d_Gnew[row * colsize + col] = (d_G[row * colsize + (col + 1)]+d_G[(row + 1) * colsize + col])/2;
			d_Bnew[row * colsize + col] = (d_B[row * colsize + (col + 1)]+d_B[(row + 1) * colsize + col])/2;
		}
		else if (row==0 &&col==(colsize-1)){
			d_Rnew[row * colsize + col] = (d_R[row * colsize + (col - 1)]+d_R[(row + 1) * colsize + col])/2;
			d_Gnew[row * colsize + col] = (d_G[row * colsize + (col - 1)]+d_G[(row + 1) * colsize + col])/2;
			d_Bnew[row * colsize + col] = (d_B[row * colsize + (col - 1)]+d_B[(row + 1) * colsize + col])/2;
		}
		else if (row==(rowsize-1) &&col==0){
			d_Rnew[row * colsize + col] = (d_R[row * colsize + (col + 1)]+d_R[(row - 1) * colsize + col])/2;
			d_Gnew[row * colsize + col] = (d_G[row * colsize + (col + 1)]+d_G[(row - 1) * colsize + col])/2;
			d_Bnew[row * colsize + col] = (d_B[row * colsize + (col + 1)]+d_B[(row - 1) * colsize + col])/2;
		}
		else if (row==(rowsize-1) &&col==(colsize-1)){
			d_Rnew[row * colsize + col] = (d_R[row * colsize + (col - 1)]+d_R[(row - 1) * colsize + col])/2;
			d_Gnew[row * colsize + col] = (d_G[row * colsize + (col - 1)]+d_G[(row - 1) * colsize + col])/2;
			d_Bnew[row * colsize + col] = (d_B[row * colsize + (col - 1)]+d_B[(row - 1) * colsize + col])/2;
		}
	}
}

// Kernel Function handles second nested for loop updates RGB values to new calculated values
__global__ void kernelCopy(int *d_Rnew, int *d_Gnew, int *d_Bnew, int *d_R, int *d_G, int *d_B, int rowsize, int colsize) {
	// Set-up
	int row = blockIdx.y*blockDim.y+threadIdx.y;
    	int col = blockIdx.x*blockDim.x+threadIdx.x;
	if (col < colsize && row < rowsize) {
		d_R[row * colsize + col] = d_Rnew[row * colsize + col];
		d_G[row * colsize + col] = d_Gnew[row * colsize + col];
		d_B[row * colsize + col] = d_Bnew[row * colsize + col];

	}
}

void performBlurs(int *h_R, int *h_G, int *h_B, int *h_Rnew, int *h_Gnew, int *h_Bnew, int rowsize, int colsize, int nblurs) {
	// Assign Memory on GPU
	// Step 1 Assign Memory on GPU
	int k;
	int sizei = sizeof(int)*rowsize*colsize;
	int *d_R, *d_G, *d_B, *d_Rnew, *d_Gnew, *d_Bnew;

	struct timeval tim;
	gettimeofday(&tim, NULL);
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	cudaMalloc((void **)&d_R,sizei);
	cudaMalloc((void **)&d_G,sizei);
	cudaMalloc((void **)&d_B,sizei);
	cudaMalloc((void **)&d_Rnew,sizei);
	cudaMalloc((void **)&d_Gnew,sizei);
	cudaMalloc((void **)&d_Bnew,sizei);

	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("Assigning Memory to GPU > %.6lf seconds elapsed\n", t2-t1);

	// Transfer to Device
	gettimeofday(&tim, NULL);
	t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	cudaMemcpy(d_R, h_R, sizei, cudaMemcpyHostToDevice);
	cudaMemcpy(d_G, h_G, sizei, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizei, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Rnew, h_Rnew, sizei, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Gnew, h_Gnew, sizei, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Bnew, h_Bnew, sizei, cudaMemcpyHostToDevice);

	t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("Transferring from host to device memory > %.6lf seconds elapsed\n", t2-t1);

	// Set up Blocks
	dim3 dimGrid(ceil(colsize/(float)BW), ceil(rowsize/(float)BH), 1);
	dim3 dimBlock(BW,BH);

	nblurs = 10; // Modify as Needed
	gettimeofday(&tim, NULL);
	t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	for (k = 0; k < nblurs; ++k) {
		kernelBlur<<<dimGrid, dimBlock>>>(d_Rnew, d_Gnew, d_Bnew, d_R, d_G, d_B, rowsize, colsize);
		kernelCopy<<<dimGrid, dimBlock>>>(d_Rnew, d_Gnew, d_Bnew, d_R, d_G, d_B, rowsize, colsize);
	}
	t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("Blurring Operation > %.6lf seconds elapsed\n", t2-t1);

	// Step 4 output copied from GPU to Host get the RGB values
	cudaMemcpy(h_R, d_R, sizei, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_G, d_G, sizei, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B, d_B, sizei, cudaMemcpyDeviceToHost);

	// Step 5 Free Memory
	cudaFree(d_R); cudaFree(d_G); cudaFree(d_B); cudaFree(d_Rnew); cudaFree(d_Gnew); cudaFree(d_Bnew);
}

int main (int argc, const char * argv[]) {
	// Assignment of initial Variables
	static int const maxlen = 200, rowsize = 521, colsize = 428, linelen = 12;
	static char str[200], lines[5][200];
	FILE *fp, *fout;
	int nlines = 0;
	unsigned int h1, h2, h3;
	char *sptr;

	// Define Host Arrays
	int *h_R, *h_G, *h_B;
	int *h_Rnew, *h_Gnew, *h_Bnew;
	int size = sizeof(int) * rowsize * colsize;
	h_R = (int *)malloc(size);
	h_G = (int *)malloc(size);
	h_B = (int *)malloc(size);
	h_Rnew = (int *)malloc(size);
	h_Gnew = (int *)malloc(size);
	h_Bnew = (int *)malloc(size);

	// Allocate Overall Size of ROw

	int row = 0, col = 0, nblurs = 0, lineno=0, k;

	// Read input file
	struct timeval tim;
	gettimeofday(&tim, NULL);
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	fp = fopen("sample.ps", "r");
	while(! feof(fp))
	{
		fscanf(fp, "\n%[^\n]", str);
		if (nlines < 5) {strcpy((char *)lines[nlines++],(char *)str);}
		else{
			for (sptr=&str[0];*sptr != '\0';sptr+=6){
				sscanf(sptr,"%2x",&h1);
				sscanf(sptr+2,"%2x",&h2);
				sscanf(sptr+4,"%2x",&h3);

				if (col==colsize){
					col = 0;
					row++;
				}
				if (row < rowsize) {
					h_R[row * colsize + col] = h1;
					h_G[row * colsize + col] = h2;
					h_B[row * colsize + col] = h3;
				}
				col++;
			}
		}
	}
	fclose(fp);
	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("Reading Input File > %.6lf seconds elapsed\n", t2-t1);

	// Run Code
	performBlurs(h_R, h_G, h_B, h_Rnew, h_Gnew, h_Bnew, rowsize, colsize, nblurs);

	gettimeofday(&tim, NULL);
	t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	fout= fopen("sampleBlurCU.ps", "w");
	for (k=0;k<nlines;k++) fprintf(fout,"\n%s", lines[k]);
	fprintf(fout,"\n");
	for(row=0;row<rowsize;row++){
		for (col=0;col<colsize;col++){
			fprintf(fout,"%02x%02x%02x",h_R[row * colsize + col],h_G[row * colsize + col],h_B[row * colsize + col]);
			lineno++;
			if (lineno==linelen){
				fprintf(fout,"\n");
				lineno = 0;
			}
		}
	}
	gettimeofday(&tim, NULL);
	t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("Outputting File > %.6lf seconds elapsed\n", t2-t1);
	fclose(fout);
    return 0;
}
