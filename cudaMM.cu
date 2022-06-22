#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>

typedef struct MatrixShape_t {
	uint32_t height;
	uint32_t width;
} MatrixShape;

uint64_t matrixSize (const MatrixShape & t) {
        uint64_t size =  ( (uint64_t)t.height * t.width );
        return size;
}


int makeMatrix (float ** t, MatrixShape & shape) {


        uint64_t tensorSize = shape.height * shape.width;
        *t = (float *) malloc (tensorSize * sizeof(float));

        float * m = * t;
        uint64_t offset;

        std::random_device random_device;
        std::uniform_real_distribution<float> dist(0.0, 1.0);

	for (uint32_t rowIdx = 0; rowIdx < shape.height; ++ rowIdx) {
		for (uint32_t colIdx = 0; colIdx < shape.width; ++ colIdx) {
			offset = rowIdx*shape.width+colIdx;
			m[offset] = dist(random_device);
		}
	}
        return 0;
}


double get_dtime(void)  {
	struct timespec t;
	double dt;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);

	dt = t.tv_nsec/1000000000.0 + t.tv_sec;

	return dt;
}

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void 
MatMult_gpu (float * a, MatrixShape aShape, 
		float * b, MatrixShape bShape,
		float * c, MatrixShape & cShape) {

	extern __shared__ float blob[];

	//i row, j col
	uint32_t ctj;
	uint32_t cti;
	uint32_t j;
	uint32_t i;
	uint32_t k;
	uint32_t tile;
	uint32_t bti;
	uint32_t btj;
	uint32_t ati;
	uint32_t atj;
	uint32_t bi;
	uint32_t bj;
	uint32_t ai;
	uint32_t aj;
	
	float *at = &blob[0];
	float *bt = &blob[32*32];

	float sum = 0;
	bool include;

	cti = blockDim.y * blockIdx.y;
	ctj = blockDim.x * blockIdx.x;
	
	ati = cti;
	atj = 0;
	bti = 0;
	btj = ctj;

	i = threadIdx.y;
	j = threadIdx.x;

	include = cti+i<aShape.height && ctj+j<bShape.width;
	for (tile = 0; tile < aShape.width; tile += 32) {
		//load
		ai = ati+i;
		aj = atj+tile+j;
		bi = bti+tile+i;
		bj = btj+j;
		
		if (ai<aShape.height && aj<aShape.width)
			at[i*32 + j] = a[ai*aShape.width + aj];
//		else
//			at[i*32 + j] = 0.0;
		if (bi<bShape.height && bj<bShape.width)
			bt[i*32 + j] = b[bi*bShape.width + bj];
//		else
//			bt[i*32 + j] = 0.0;
		__syncthreads();
		
		//acummalate
		if (include) {
			for (k = 0; k < 32; k++) {
				aj = atj+tile+k;
				bi = bti+tile+k;
				if (aj < aShape.width && bi < bShape.height)
					sum += at[i*32+k] * bt[k*32+j];
			}
		}

		__syncthreads();
	}
	
	//output
	if (include)
		c[(cti+i)*bShape.width + ctj+j] = sum;
}



int evaluateMMgpu (MatrixShape aShape, MatrixShape bShape, MatrixShape & cShape) {

	float *a = NULL;
	float *b = NULL;
	float *c = NULL;
	
        MatrixShape *cShape_d;
	float *a_d;
	float *b_d;
	float *c_d;
	
	cShape.height = aShape.height;
	cShape.width = bShape.width;

	cudaFree(NULL);

        int retVal;
        retVal = makeMatrix(&a, aShape);
        retVal = makeMatrix(&b, bShape);
        retVal = makeMatrix(&c, cShape);

	cudaMalloc(&cShape_d, sizeof(*cShape_d));
	cudaMalloc(&c_d, sizeof(float)*matrixSize(cShape));
	cudaMalloc(&a_d, sizeof(float)*matrixSize(aShape));
	cudaMalloc(&b_d, sizeof(float)*matrixSize(bShape));

        cudaMemcpy(a_d, a, sizeof(float)*matrixSize(aShape), cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, sizeof(float)*matrixSize(bShape), cudaMemcpyHostToDevice);
        cudaMemcpy(cShape_d, &cShape, sizeof(cShape), cudaMemcpyHostToDevice);

/*
	cudaMallocManaged(&a, aShape.height*aShape.width*aShape.channels*aShape.count*sizeof(float));
	cudaMallocManaged(&b, bShape.height*bShape.width*bShape.channels*bShape.count*sizeof(float));
	cudaMallocManaged(&c, cShape.height*cShape.width*cShape.channels*cShape.count*sizeof(float));
	cudaMallocManaged(&cShape_d, sizeof(*cShape_d));
*/

        dim3 blockSize(32,32,1);
        dim3 gridSize((bShape.width+31)/32,(aShape.height+31)/32,1);

	//kernel
	MatMult_gpu<<<gridSize, blockSize, 2*32*32*sizeof(float)>>>(a_d,aShape,b_d,bShape,c_d,*cShape_d);
	//MatMult_gpu<<<gridSize, blockSize, 2*32*32*sizeof(float)>>>(a,aShape,b,bShape,c,*cShape_d);

        cudaMemcpy(c, c_d, sizeof(float)*matrixSize(cShape), cudaMemcpyDeviceToHost);

	return 0;
}

int runMMgpu (int argc, char ** argv) {

	MatrixShape aShape = {4096, 4096};
        MatrixShape bShape = {4096, 4096};
        MatrixShape cShape;

	evaluateMMgpu (aShape, bShape, cShape);
	return 0;
}


int main(int argc, char *argv[])
{
	runMMgpu(argc, argv);
}
