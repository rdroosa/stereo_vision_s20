#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <queue>

#define DEBUG_(x) (std::cout << "DEBUG: " << x << std::endl);
#define ROWS 480
#define COLS 640

using namespace cv;

__host__ __device__ int rel_i (int index, int d_cols, int d_rows)
{
	return (index + (d_rows * COLS) + d_cols) % (COLS * ROWS);
}

__global__ void k_intensity (uchar *img_in, uchar *intensity)
{
   register unsigned int i_index =  (blockIdx.x * 3 * 1024) + 3 * threadIdx.x;
   register unsigned int o_index = (1024 * blockIdx.x) + threadIdx.x;

   intensity[o_index] = (img_in[i_index]*0.114 + img_in[i_index + 1]*0.587 + img_in[i_index + 2]*0.2989); 
}

__global__ void k_gaussian (uchar *in, uchar *out)
{
	register int index = (1024 * blockIdx.x) + threadIdx.x;	
	register int mod = index % 640;
	if ((index > 639) && (index < 306560) && (mod != 0) && (mod != 639))
	{
		out[index] = 
			(in[rel_i(index, -1, -1)] + 2*in[rel_i(index, 0, -1)] + in[rel_i(index, 1, -1)]
			+ 2*in[rel_i(index, -1, 0)] + 4*in[rel_i(index, 0, 0)] + 2*in[rel_i(index, 1, -1)]
			+ in[rel_i(index, -1, 1)] + 2*in[rel_i(index, 0, 1)] + in[rel_i(index, 1, 1)]) / 16;
	}
} 

__global__ void k_sobel (uchar *intensity, uchar *sobel, int high, int low)
{
	register int index = (1024 * blockIdx.x) + threadIdx.x;
	register int mod = index % 640;

	if ((index > 639) && (index < 306560) && (mod != 0) && (mod != 639)) 
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
	
	register int index = (1024 * blockIdx.x) + threadIdx.x;
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
	register int index = (1024 * blockIdx.x) + threadIdx.x;
	in[index] = in[index] * (in[index] == 255);
}

__host__ void c_hyst(uchar *img, int n_pix)
{
	std::queue<int> hyst_queue;
	int test_index;
	for (int i = 0; i < n_pix; i++)
	{
		if (img[i] == 255)
		{
			for (int dx = -1; dx < 2; dx++)
			{
				for(int dy = -1; dy < 2; dy++)
				{
					test_index = rel_i(i, dx, dy);
					if (img[test_index] == 128)
					{ 
						img[test_index] = 255;
						hyst_queue.push(test_index);
					}
				}
			}
		}
	}
	
	while(!hyst_queue.empty())
	{
		int i = hyst_queue.front();
		for (int dx = -1; dx < 2; dx++)
		{
			for(int dy = -1; dy < 2; dy++)
			{
				test_index = rel_i(i, dx, dy);
				if (img[test_index] == 128)
				{ 
					img[test_index] = 255;
					hyst_queue.push(test_index);
				}
			}
		}
		hyst_queue.pop();
	}
}


int main(void)
{
    cudaEvent_t start_t, intensity_t, gaussian_t, sobel_t, hyst_t, stop_t;
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);

	Mat img = imread("test_data/Backpack-perfect/im0.png", IMREAD_COLOR);
	Mat scale_img;

	Size scale_size = Size(COLS, ROWS);

	resize(img, scale_img, scale_size, 0, 0);
	imwrite("scaled.png", scale_img);

    int n_pix = scale_img.rows * scale_img.cols;
    int n_subpix = n_pix * scale_img.channels();
	char *done;
	int *weak_indices;
	int n_weak;

    uchar *img_in, *intensity, *sobel, *gaussian1, *gaussian2;
    
    cudaMallocManaged(&img_in, n_subpix * sizeof(img.data[0]));
    cudaMallocManaged(&intensity, n_pix * sizeof(img.data[0]));
	cudaMallocManaged(&gaussian1, n_pix * sizeof(img.data[0]));
	cudaMallocManaged(&gaussian2, n_pix * sizeof(img.data[0]));
	cudaMallocManaged(&sobel, n_pix * sizeof(img.data[0]));
	cudaMallocManaged(&weak_indices, n_pix * sizeof(int));
	cudaMallocManaged(&done, sizeof(char)); 

    for (int i=0; i < n_subpix; i++)
    {
        img_in[i] = scale_img.data[i];
    }

	cudaDeviceSynchronize();
	*done = 0;
	cudaDeviceSynchronize();
	int n_runs = 100;
	float ms;
	int traverse_steps = 0; 
	float running = 0;


	for (int i = 0; i < n_runs; i++)
	{
		cudaEventRecord(start_t);
		k_intensity<<<300, 1024>>>(img_in, intensity);	
		k_gaussian<<<300, 1024>>>(intensity, gaussian1);
		k_gaussian<<<300, 1024>>>(gaussian1, gaussian2);
		k_sobel<<<300, 1024>>>(gaussian2, sobel, 100, 40);
		
		cudaDeviceSynchronize();
		c_hyst(sobel, n_pix);
		cudaEventRecord(stop_t);
		cudaEventSynchronize(stop_t);
		cudaDeviceSynchronize(); 
		cudaEventElapsedTime(&ms, start_t, stop_t);

		running += ms;
	}
	DEBUG_("TRAVERSE: " << traverse_steps)
    DEBUG_("AVERAGE: " << running / n_runs << std::endl << "LAST: " << ms) 

	img = Mat(scale_img.rows, scale_img.cols, CV_8UC1, sobel);
    DEBUG_("READ IMGS")
    imwrite("out.png", img);

    return 0;
}
