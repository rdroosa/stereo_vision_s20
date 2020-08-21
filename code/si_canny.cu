#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#define DEBUG_(x) (std::cout << "DEBUG: " << x << std::endl);
#define MOD (2016 * 2940 * 3)

using namespace cv;

__global__ 
void k_canny (uchar *img, uchar *canny)
{
	// Calculate indices
	register char is_img_edge = 
			((blockIdx.x == 0) * 0b1000)
		|	((blockIdx.x == (gridDim.x-1)) * 0b0100)
		|	((blockIdx.y == 0) * 0b0010)
		|	((blockIdx.y == (gridDim.y-1)) * 0b0001);
							  
	register int thread_id = blockDim.x * threadIdx.y + threadIdx.x;
	
	register int load_y1 = thread_id / (blockDim.x + 2);
	register int load_x1 = thread_id % (blockDim.x + 2);
	
	register int load_y2 = (960 + thread_id) / (blockDim.x + 2);
	register int load_x2 = (960 + thread_id) % (blockDim.x + 2);
	
	register int i_img1 = 8820 * (32 * blockIdx.y + load_y1 - 1) + 3 * (30 * blockIdx.x + load_x1 - 1);
	register int i_img2 = 8820 * (32 * blockIdx.y + load_y2 - 1) + 3 * (30 * blockIdx.x + load_x2 - 1);
	register int i_canny = 2940 * (32 * blockIdx.y + threadIdx.y) + (30 * blockIdx.x + threadIdx.x);
	register int i_tx = threadIdx.x + 1;
	register int i_ty = threadIdx.y + 1;
	
	register char not_img_edge1 = 1;
	register char not_img_edge2 = 1;
	
	if (is_img_edge)
	{
		not_img_edge1 = 
				((((load_x1 == 0) * 0b1000)
			|	((load_x1 == (blockDim.x-1)) * 0b0100)
			|	((load_y1 == 0) * 0b0010)
			|	((load_y1 == (blockDim.y-1)) * 0b0001))
			&	is_img_edge) == 0;
			
		not_img_edge2 = 
				((((load_x2 == 0) * 0b1000)
			|	((load_x2 == (blockDim.x-1)) * 0b0100)
			|	((load_y2 == 0) * 0b0010)
			|	((load_y2 == (blockDim.y-1)) * 0b0001))
			&	is_img_edge) == 0;
	}
	
	// Declare utiliy variables
	register int v, h;        					// Gradient vector components 
	register char dir, diag, is_max;			// Non-maximal suppression boolean variables
	register float dir_weight, diag_weight; 	// Non-maximal suppression interpolation weights

	// Allocate shared memory tiles
	__shared__ uchar intensity	[32][34];
	__shared__ uchar sobel		[32][34];
	__shared__ uchar nms		[32][34];
	
	// Calculate intensity and load tile into shared memory
	if (not_img_edge1)
	{
		intensity[load_x1][load_y1] = (img[i_img1] + img[i_img1 + 1] + img[i_img1 + 2]) / 3;
	}
	else
	{
		intensity[load_x1][load_y1] = 0;
	}
		
	if (((960 + thread_id) < 1088) && (not_img_edge2))
	{
		intensity[load_x2][load_y2] = (img[i_img2] + img[i_img2 + 1] + img[i_img2 + 2]) / 3;
	}
	else
	{
		intensity[load_x2][load_y2] = 0;
	}
			
	__syncthreads();

	// Compute sobel operator over shared memory tile
	v =   intensity		[i_tx-1] [i_ty-1] 
		+ 2*intensity	[i_tx]   [i_ty-1] 
		+ intensity		[i_tx+1] [i_ty-1]
		- intensity		[i_tx-1] [i_ty+1]  
		- 2*intensity	[i_tx]   [i_ty+1] 
		- intensity		[i_tx+1] [i_ty+1];
		
	h =   intensity		[i_tx+1] [i_ty-1]
		+ 2*intensity	[i_tx+1] [i_ty] 
		+ intensity		[i_tx+1] [i_ty+1]
		- intensity		[i_tx-1] [i_ty-1] 
		- 2*intensity	[i_tx-1] [i_ty] 
		- intensity		[i_tx-1] [i_ty+1];
		
	sobel[i_ty][i_ty] = (uchar) sqrt((float) (v*v + h*h));   	
	
	__syncthreads();
	
	// Apply non-maximal suppression and threshold min/maxing over shared memory tile
	dir = abs(h) > abs(v);
	diag = (( h > 0 ) == (v > 0));	
	
	dir_weight = dir * ((float) v) / ((float) h);
	diag_weight = 1 - dir_weight;
	
	is_max = 
			(sobel[i_tx][i_ty] > 
				(dir_weight * sobel[i_tx + dir][i_ty + !dir]
				+ diag_weight * sobel[i_tx + diag - !diag][i_ty + 1]))
		&
			(sobel[i_tx][i_ty] >
				(dir_weight * sobel[i_tx - dir][i_ty - !dir]
				+ diag_weight * sobel[i_tx - diag + !diag][i_ty - 1]));	
				
	nms[i_tx][i_ty] = 
		is_max * 
		(
				(128 * (sobel[i_tx][i_ty] > 200))
			+ 	
				(127 * (sobel[i_tx][i_ty] > 200))
		);
	
	__syncthreads();
	
	// Apply local hysteresis edge detection
	
	// Copy shared memory tile to global memory
	canny[i_canny] = intensity[i_tx][i_ty];
}

int main(void)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Mat img = imread("test_data/Backpack-perfect/im0.png", IMREAD_COLOR);

	namedWindow("canny", WINDOW_AUTOSIZE);

    int n_pix = img.rows * img.cols;
    int n_subpix = n_pix * img.channels();

    uchar *img_in, *canny;
    
    cudaMallocManaged(&img_in, n_subpix * sizeof(img.data[0]));
	cudaMallocManaged(&canny, n_pix * sizeof(img.data[0]));

    for (int i=0; i < n_subpix; i++)
    {
        img_in[i] = img.data[i];
    }
    
    dim3 block(30, 32, 1);
    dim3 grid(98, 63, 1);
    
    float ms;
    float running = 0;
	
	int n_runs = 100;	
	    
    for (int i = 0; i < n_runs; i++)
    {
    	cudaEventRecord(start);
    	k_canny<<<grid, block>>>(img_in, canny);
		cudaEventRecord(stop);
    	cudaEventSynchronize(stop);
    	cudaDeviceSynchronize();
    	cudaEventElapsedTime(&ms, start, stop);
		running += ms; 
    }
    
    DEBUG_("AVERAGE: " << running / n_runs << std::endl << "LAST: " << ms)
	
	Mat img_out = Mat(img.rows, img.cols, CV_8UC1, canny);
    DEBUG_("READ IMG")
    imshow("canny", img_out);
    waitKey(0);
	
	/*
	clock_t startt, stopt;
	Mat gray, cannyimg;
	startt = clock();
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	cv::Canny(gray, cannyimg, 100, 200);
	stopt = clock();
	double runtime = double(stopt - startt) / double(CLOCKS_PER_SEC);
	DEBUG_("CPU TIME: " << runtime)
	imshow("canny", cannyimg);
	waitKey(0);   
	*/
	
    return 0;
}
