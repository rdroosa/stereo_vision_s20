#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#define DEBUG_(x) (std::cout << "DEBUG: " << x << std::endl);
#define MOD (2016 * 2940 * 3)
#define ROWS 2016
#define COLS 2940

using namespace cv;

__device__ int rel_i (int index, int d_cols, int d_rows)
{
	return (index + (d_rows * COLS) + d_cols) % (COLS * ROWS);
}

__global__ void k_intensity (uchar *img_in, uchar *intensity)
{
   register unsigned int i_index =  (blockIdx.x * 2880) + 3 * threadIdx.x;
   register unsigned int o_index = (960 * blockIdx.x) + threadIdx.x;

   intensity[o_index] = (img_in[i_index]*0.114 + img_in[i_index + 1]*0.587 + img_in[i_index + 2]*0.2989); 
}

__global__ void k_gaussian (uchar *in, uchar *out)
{
	register int index = (960 * blockIdx.x) + threadIdx.x;	
	register int mod = index % 5927040;
	if ((index > 2939) && (index < 5924200) && (mod != 0) && (mod != 2939))
	{
		out[index] = 
			(in[rel_i(index, -1, -1)] + 2*in[rel_i(index, 0, -1)] + in[rel_i(index, 1, -1)]
			+ 2*in[rel_i(index, -1, 0)] + 4*in[rel_i(index, 0, 0)] + 2*in[rel_i(index, 1, -1)]
			+ in[rel_i(index, -1, 1)] + 2*in[rel_i(index, 0, 1)] + in[rel_i(index, 1, 1)]) / 16;
	}
} 

__global__ void k_sobel (uchar *intensity, uchar *sobel, int high, int low)
{
	register int index = (960 * blockIdx.x) + threadIdx.x;
	register int mod = index % 5927040;

	if ((index > 2939) && (index < 5924200) && (mod != 0) && (mod != 2939)) 
	{
    	register int h, v, mag;
		register char dir, diag, is_max;
		float dir_weight, diag_weight;
		v = intensity[rel_i(index, -1, 1)] + 2 * intensity[rel_i(index, 0, 1)] +  intensity[rel_i(index, 1, 1)] 
			- intensity[rel_i(index, -1, -1)] - 2 * intensity[rel_i(index, 0, -1)] - intensity[rel_i(index, 1, -1)];

		h = intensity[rel_i(index, 1, -1)] + 2 * intensity[rel_i(index, 1, 0)] +  intensity[rel_i(index, 1, 1)]
			- intensity[rel_i(index, -1, -1)] - 2 * intensity[rel_i(index, -1, 0)] - intensity[rel_i(index, -1, 1)];

		mag = v * v + h * h;

		dir = abs(h) > abs(v);
		diag = (( h > 0 ) == ( v > 0 ));	

		dir_weight = dir * ((float) v) / ((float) h);
		diag_weight = 1 - dir_weight;

		mag = sqrt((float) mag);
		sobel[index] = mag;

		__syncthreads();

		is_max = 
				(mag > 
					(dir_weight * sobel[rel_i(index, dir, !dir)]
					+ diag_weight * sobel[rel_i(index, diag - !diag,  1)]))
			&&
				(mag >
					(dir_weight * sobel[rel_i(index, -dir, -(!dir))]
					+ diag_weight * sobel[rel_i(index, !diag - diag, -1)]));

		__syncthreads();
		
		sobel[index] = is_max * 128 * (mag > low) + 127 * (mag > high);	
	}
}

__global__ void k_hyst_traverse (uchar *in, char *done)
{
	
	register int index = (960 * blockIdx.x) + threadIdx.x;
	if (in[index] == 128)
	{
		in[index] += 127 * (
			(in[rel_i(index, -1, -1)] == 255) ||
			(in[rel_i(index, -1,  0)] == 255) ||
			(in[rel_i(index, -1,  1)] == 255) ||
			(in[rel_i(index,  0, -1)] == 255) ||
			(in[rel_i(index,  0,  1)] == 255) ||
			(in[rel_i(index,  1, -1)] == 255) ||
			(in[rel_i(index,  1,  0)] == 255) ||
			(in[rel_i(index,  1,  1)] == 255)
		);
		if (in[index] == 255)
		{
			*done = 0;
		}
	}
	
}

__global__ void k_hyst_prune(uchar *in)
{
	register int index = (960 * blockIdx.x) + threadIdx.x;
	in[index] = in[index] * (in[index] == 255);
}

int main(void)
{
    cudaEvent_t start_t, intensity_t, gaussian_t, sobel_t, hyst_t, stop_t;
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);

    Mat img = imread("test_data/Backpack-perfect/im0.png", IMREAD_COLOR);
	namedWindow("canny", WINDOW_NORMAL);

    int n_pix = img.rows * img.cols;
    int n_subpix = n_pix * img.channels();
	char *done;

    uchar *img_in, *intensity, *sobel, *gaussian;
    
    cudaMallocManaged(&img_in, n_subpix * sizeof(img.data[0]));
    cudaMallocManaged(&intensity, n_pix * sizeof(img.data[0]));
	cudaMallocManaged(&gaussian, n_pix * sizeof(img.data[0]));
	cudaMallocManaged(&sobel, n_pix * sizeof(img.data[0]));
	cudaMallocManaged(&done, sizeof(char)); 

    for (int i=0; i < n_subpix; i++)
    {
        img_in[i] = img.data[i];
    }

	cudaDeviceSynchronize();
	*done = 0;
	cudaDeviceSynchronize();
	int n_runs = 10;
	float ms;
	int traverse_steps = 0; 
	float running = 0;


	for (int i = 0; i < n_runs; i++)
	{
		cudaEventRecord(start_t);
		k_intensity<<<6174, 960>>>(img_in, intensity);	
		k_gaussian<<<6174, 960>>>(intensity, gaussian);
		k_sobel<<<6174, 960>>>(gaussian, sobel, 70, 60);
		cudaDeviceSynchronize();
		cudaEventRecord(stop_t);
		cudaEventSynchronize(stop_t);
		while (!*done)
		{	
			traverse_steps += 1;
			*done = 1;
			k_hyst_traverse<<<6174, 960>>>(sobel, done);
			cudaDeviceSynchronize();
		}
		k_hyst_prune<<<6174, 960>>>(sobel);
		
		cudaDeviceSynchronize(); 
		cudaEventElapsedTime(&ms, start_t, stop_t);

		running += ms;
	}
	DEBUG_("TRAVERSE: " << traverse_steps)
    DEBUG_("AVERAGE: " << running / n_runs << std::endl << "LAST: " << ms) 

	img = Mat(img.rows, img.cols, CV_8UC1, sobel);
    DEBUG_("READ IMGS")
    imshow("canny", img);
    waitKey(0);

    return 0;
}
