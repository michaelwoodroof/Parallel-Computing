#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <omp.h>

#define CHUNKSIZE 200 // Change as Needed

int main (int argc, const char * argv[]) {
	static int const maxlen = 200, rowsize = 521, colsize = 428, linelen = 12;
	char str[maxlen], lines[5][maxlen];
	FILE *fp, *fout;
	int nlines = 0;
	unsigned int h1, h2, h3;
	char *sptr;
	static int R[521][428], G[521][428], B[521][428];
	static int Rnew[521][428], Gnew[521][428], Bnew[521][428];
	int row = 0, col = 0, nblurs, lineno=0, k;
	struct timeval tim;

	fp = fopen("David.ps", "r");

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
					R[row][col] = h1;
					G[row][col] = h2;
					B[row][col] = h3;
				}
				col++;
			}
		}
	}
	fclose(fp);

	// OpenMP
	int chunk;
	static int m[521][428];
	static float a[521][428], b[521][428], c[521][428];
	for(row = 0; row < rowsize; row++){
		for (col = 0; col < colsize; col++){
			a[row][col] = (row * colsize + col) * 1.0;
			b[row][col] = (row + col * colsize) * 1.0;
		}
	}
	chunk = CHUNKSIZE;
	omp_set_num_threads(6); // Modify as Needed

	nblurs = 10;
	gettimeofday(&tim, NULL);
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	for (k = 0; k < nblurs; k++){
		#pragma omp parallel shared(a, b, c, chunk) private(row, col)
		{
			#pragma omp for collapse(2) schedule(dynamic, chunk) nowait
				for (row = 0;row < rowsize; row++) {
					for (col = 0; col < colsize; col++) {
						c[row][col] = a[row][col] + b[row][col];
						m[row][col] = omp_get_thread_num();
						if (row != 0 && row != (rowsize-1) && col != 0 && col != (colsize-1)){
							Rnew[row][col] = (R[row+1][col]+R[row-1][col]+R[row][col+1]+R[row][col-1])/4;
							Gnew[row][col] = (G[row+1][col]+G[row-1][col]+G[row][col+1]+G[row][col-1])/4;
							Bnew[row][col] = (B[row+1][col]+B[row-1][col]+B[row][col+1]+B[row][col-1])/4;
						}
						else if (row == 0 && col != 0 && col != (colsize-1)) {
							Rnew[row][col] = (R[row+1][col]+R[row][col+1]+R[row][col-1])/3;
							Gnew[row][col] = (G[row+1][col]+G[row][col+1]+G[row][col-1])/3;
							Bnew[row][col] = (B[row+1][col]+B[row][col+1]+B[row][col-1])/3;
						}
						else if (row == (rowsize-1) && col != 0 && col != (colsize-1)){
							Rnew[row][col] = (R[row-1][col]+R[row][col+1]+R[row][col-1])/3;
							Gnew[row][col] = (G[row-1][col]+G[row][col+1]+G[row][col-1])/3;
							Bnew[row][col] = (B[row-1][col]+B[row][col+1]+B[row][col-1])/3;
						}
						else if (col == 0 && row != 0 && row != (rowsize-1)){
							Rnew[row][col] = (R[row+1][col]+R[row-1][col]+R[row][col+1])/3;
							Gnew[row][col] = (G[row+1][col]+G[row-1][col]+G[row][col+1])/3;
							Bnew[row][col] = (B[row+1][col]+B[row-1][col]+B[row][col+1])/3;
						}
						else if (col == (colsize-1) && row != 0 && row != (rowsize-1)){
							Rnew[row][col] = (R[row+1][col]+R[row-1][col]+R[row][col-1])/3;
							Gnew[row][col] = (G[row+1][col]+G[row-1][col]+G[row][col-1])/3;
							Bnew[row][col] = (B[row+1][col]+B[row-1][col]+B[row][col-1])/3;
						}
						else if (row==0 &&col==0){
							Rnew[row][col] = (R[row][col+1]+R[row+1][col])/2;
							Gnew[row][col] = (G[row][col+1]+G[row+1][col])/2;
							Bnew[row][col] = (B[row][col+1]+B[row+1][col])/2;
						}
						else if (row==0 &&col==(colsize-1)){
							Rnew[row][col] = (R[row][col-1]+R[row+1][col])/2;
							Gnew[row][col] = (G[row][col-1]+G[row+1][col])/2;
							Bnew[row][col] = (B[row][col-1]+B[row+1][col])/2;
						}
						else if (row==(rowsize-1) &&col==0){
							Rnew[row][col] = (R[row][col+1]+R[row-1][col])/2;
							Gnew[row][col] = (G[row][col+1]+G[row-1][col])/2;
							Bnew[row][col] = (B[row][col+1]+B[row-1][col])/2;
						}
						else if (row==(rowsize-1) &&col==(colsize-1)){
							Rnew[row][col] = (R[row][col-1]+R[row-1][col])/2;
							Gnew[row][col] = (G[row][col-1]+G[row-1][col])/2;
							Bnew[row][col] = (B[row][col-1]+B[row-1][col])/2;
						}
					}
				}
		}

		for(row=0;row<rowsize;row++){
			for (col=0;col<colsize;col++){
			    R[row][col] = Rnew[row][col];
			    G[row][col] = Gnew[row][col];
			    B[row][col] = Bnew[row][col];
			}
		}
	}
	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed\n", t2-t1);

	fout= fopen("DavidBlurOMS.ps", "w");
	for (k=0;k<nlines;k++) fprintf(fout,"\n%s", lines[k]);
	fprintf(fout,"\n");
	for(row=0;row<rowsize;row++){
		for (col=0;col<colsize;col++){
			fprintf(fout,"%02x%02x%02x",R[row][col],G[row][col],B[row][col]);
			lineno++;
			if (lineno==linelen){
				fprintf(fout,"\n");
				lineno = 0;
			}
		}
	}
	fclose(fout);
    return 0;
}
