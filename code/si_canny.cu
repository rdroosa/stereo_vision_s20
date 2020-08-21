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
	// Calculate indices, alloc shared memory tiles
	register int row = 28 * blockIdx.y + threadIdx.y - 1;
	register int col = 30 * blockIdx.x + threadIdx.x - 1;
	
	register int i_img = 8820 * row + 3 * col;
	register int i_canny = 2940 * row + col;
	
	register int not_tile_edge = !((threadIdx.x == 0) || (threadIdx.y == 0) || (threadIdx.x == 31) || (threadIdx.y == 29)); 
	register int not_img_edge = ((row > 0) && (row < 2016) && (col > 0) && (col < 2940));
	register int v, h;
	register char dir, diag, is_max;
	register float dir_weight, diag_weight; 
	
	__shared__ uchar intensity	[32][30];
	__shared__ uchar sobel		[32][30];
	__shared__ uchar nms		[32][30];
	
	// Calculate intensity and load tile into shared memory
	if (not_img_edge)
	{
		intensity[threadIdx.x][threadIdx.y] = (img[i_img] + img[i_img + 1] + img[i_img + 2]) / 3;
	}
	__syncthreads();

	// Compute sobel operator over shared memory tile
	if (not_tile_edge)
	{
		v =   intensity		[threadIdx.x-1][threadIdx.y-1] 
			+ 2*intensity	[threadIdx.x][threadIdx.y-1] 
			+ intensity		[threadIdx.x+1][threadIdx.y-1]
			- intensity		[threadIdx.x-1][threadIdx.y+1]  
			- 2*intensity	[threadIdx.x][threadIdx.y+1] 
			- intensity		[threadIdx.x+1][threadIdx.y+1];
			
		h =   intensity		[threadIdx.x+1][threadIdx.y-1]
			+ 2*intensity	[threadIdx.x+1][threadIdx.y] 
			+ intensity		[threadIdx.x+1][threadIdx.y+1]
			- intensity		[threadIdx.x-1][threadIdx.y-1] 
			- 2*intensity	[threadIdx.x-1][threadIdx.y] 
			- intensity		[threadIdx.x-1][threadIdx.y+1];
			
		sobel[threadIdx.x][threadIdx.y] = (uchar) sqrt((float) (v*v + h*h));   	
	} 
	
	__syncthreads();
	
	// Apply non-maximal suppression and threshold min/maxing over shared memory tile
	
	if (not_tile_edge)
	{
		dir = abs(h) > abs(v);
		diag = (( h > 0 ) == (v > 0));	
		
		dir_weight = dir * ((float) v) / ((float) h);
		diag_weight = 1 - dir_weight;
		
		is_max = 
				(sobel[threadIdx.x][threadIdx.y] > 
					(dir_weight * sobel[threadIdx.x + dir][threadIdx.y + !dir]
					+ diag_weight * sobel[threadIdx.x + diag - !diag][threadIdx.y + 1]))
			&
				(sobel[threadIdx.x][threadIdx.y] >
					(dir_weight * sobel[threadIdx.x - dir][threadIdx.y - !dir]
					+ diag_weight * sobel[threadIdx.x - diag + !diag][threadIdx.y - 1]));	
					
		nms[threadIdx.x][threadIdx.y] = 
			is_max * 
			(
					(128 * (sobel[threadIdx.x][threadIdx.y] > 200))
				+ 	
					(127 * (sobel[threadIdx.x][threadIdx.y] > 200))
			);
	}
	
	__syncthreads();
	
	// Apply local hysteresis edge detection
	
	// Copy shared memory tile to global memory
	if (not_img_edge & not_tile_edge)
	{
		canny[i_canny] = nms[threadIdx.x][threadIdx.y];
	}
}

int main(void)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Mat img = imread("test_data/Backpack-perfect/im0.png", IMREAD_COLOR);

	namedWindow("canny", WINDOW_NORMAL);

    int n_pix = img.rows * img.cols;
    int n_subpix = n_pix * img.channels();

    uchar *img_in, *canny;
    
    cudaMallocManaged(&img_in, n_subpix * sizeof(img.data[0]));
	cudaMallocManaged(&canny, n_pix * sizeof(img.data[0]));

    for (int i=0; i < n_subpix; i++)
    {
        img_in[i] = img.data[i];
    }
    
    dim3 block(32, 30, 1);
    dim3 grid(98, 72, 1);
    
    for (int i = 0; i < 100; i++)
    {
    	k_canny<<<grid, block>>>(img_in, canny);
    }
    
	cudaEventRecord(start);
    k_canny<<<grid, block>>>(img_in, canny);
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
   
    float ms;

    cudaEventElapsedTime(&ms, start, stop);

    DEBUG_("KERNEL TIME: " << ms) 
	
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
