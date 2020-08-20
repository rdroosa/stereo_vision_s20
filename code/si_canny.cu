#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#define DEBUG_(x) (std::cout << "DEBUG: " << x << std::endl);
#define MOD (2016 * 2940 * 3)

using namespace cv;

__global__ void k_intensity (uchar *img_in, uchar *intensity)
{
   register unsigned int i_index =  (blockIdx.x * 2880) + 3 * threadIdx.x;
   register unsigned int o_index = (960 * blockIdx.x) + threadIdx.x;

   intensity[o_index] = (img_in[i_index] + img_in[i_index + 1] + img_in[i_index + 2]) / 3; 
}

__global__ void k_sobel (uchar *intensity, uchar *sobel)
{
	register int index = (960 * blockIdx.x) + threadIdx.x;
	register int mod = index % 5927040;
    

	if ((index > 2939) && (index < 5924200) && (mod != 0) && (mod != 2939)) 
	{
    	register int h, v, mag;
		v = intensity[index - 2941] + 2 * intensity[index - 2940] +  intensity[index - 2939] 
		- intensity[index + 29410] - 2 * intensity[index + 2940] - intensity[index + 2939];

		h = intensity[index - 2941] + 2 * intensity[index - 1] +  intensity[index + 2939]
			- intensity[index + 2941] - 2 * intensity[index + 1] - intensity[index - 2939];
		
		mag = v * v + h * h;
	
		sobel[index] = sqrt((float) mag);		
	}
}


__global__ void k_hyst (uchar *sobel_h, uchar *hyster_thresh)
{

}

int main(void)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Mat img = imread("test_data/Backpack-perfect/im0.png", IMREAD_COLOR);

    namedWindow("high", WINDOW_NORMAL);
	namedWindow("low", WINDOW_NORMAL);
	namedWindow("intensity", WINDOW_NORMAL);
	namedWindow("canny", WINDOW_NORMAL);

    int n_pix = img.rows * img.cols;
    int n_subpix = n_pix * img.channels();

    uchar *img_in, *intensity, *sobel;
    
    cudaMallocManaged(&img_in, n_subpix * sizeof(img.data[0]));
    cudaMallocManaged(&intensity, n_pix * sizeof(img.data[0]));
	cudaMallocManaged(&sobel, n_pix * sizeof(img.data[0]));

    for (int i=0; i < n_subpix; i++)
    {
        img_in[i] = img.data[i];
    }

	cudaEventRecord(start);
    k_intensity<<<6174, 960>>>(img_in, intensity);
	
    //cudaDeviceSynchronize();
	
/*
    Mat imgintensity = Mat(img.rows, img.cols, CV_8UC1, intensity);
    DEBUG_("READ INT")
    imshow("intensity", imgintensity);
    waitKey(0); 
*/

    k_sobel<<<6174, 960>>>(intensity, sobel);
    cudaDeviceSynchronize(); 

	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
   
    float ms;

    cudaEventElapsedTime(&ms, start, stop);

    DEBUG_("KERNEL TIME: " << ms) 
	
	Mat high = Mat(img.rows, img.cols, CV_8UC1, sobel);
    DEBUG_("READ IMGS")
    imshow("high", high);
    waitKey(0);
	
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

    return 0;
}
