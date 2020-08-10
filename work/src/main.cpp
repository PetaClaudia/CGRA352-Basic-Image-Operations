
// std
#include <iostream>
#include <stdio.h>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// project
#include "invert.hpp"
#include "filter.hpp"

using namespace cv;
using namespace std;


// main program
// 
int main( int argc, char** argv ) {

	// check we have exactly one additional argument
	// eg. res/vgc-logo.png
	if( argc != 2) {
		cerr << "Usage: cgra352 <Image>" << endl;
		abort();
	}


	// read the file
	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR); 

	// check for invalid input
	if(!image.data ) {
		cerr << "Could not open or find the image" << std::endl;
		abort();
	}


    //image in hsv
    Mat imageHSV = convertBGR2HSV(image);
    
    //split into channels
    Mat B = getChannels(image)[0];
    Mat G = getChannels(image)[1];
    Mat R = getChannels(image)[2];
    Mat H = getChannels(imageHSV)[0];
    Mat S = getChannels(imageHSV)[1];
    Mat V = getChannels(imageHSV)[2];
    
    //channels
    std::vector<cv::Mat> ch = getChannels(imageHSV);
    Mat outputHue;

    //scale hue then merge and convert
    ch[0] = H*0.0;
    merge(ch, outputHue);
    Mat H_0 = convertHSV2BGR(outputHue);
    
    ch[0] = H*0.2;
    merge(ch, outputHue);
    Mat H_2 = convertHSV2BGR(outputHue);
    
    ch[0] = H*0.4;
    merge(ch, outputHue);
    Mat H_4 = convertHSV2BGR(outputHue);
    
    ch[0] = H*0.6;
    merge(ch, outputHue);
    Mat H_6 = convertHSV2BGR(outputHue);
    
    ch[0] = H*0.8;
    merge(ch, outputHue);
    Mat H_8 = convertHSV2BGR(outputHue);
    
    //reset
    std::vector<cv::Mat> ch2 = getChannels(imageHSV);
    Mat outputSat;
    
    //scale saturation then merge and convert
    ch2[1] = S*0.0;
    merge(ch2, outputSat);
    Mat S_0 = convertHSV2BGR(outputSat);
    
    ch2[1] = S*0.2;
    merge(ch2, outputSat);
    Mat S_2 = convertHSV2BGR(outputSat);
    
    ch2[1] = S*0.4;
    merge(ch2, outputSat);
    Mat S_4 = convertHSV2BGR(outputSat);
    
    ch2[1] = S*0.6;
    merge(ch2, outputSat);
    Mat S_6 = convertHSV2BGR(outputSat);
    
    ch2[1] = S*0.8;
    merge(ch2, outputSat);
    Mat S_8 = convertHSV2BGR(outputSat);
    
    //reset
    std::vector<cv::Mat> ch3 = getChannels(imageHSV);
    Mat outputVal;
       
    //scale value then merge and convert
    ch3[2] = V*0.0;
    merge(ch3, outputVal);
    Mat V_0 = convertHSV2BGR(outputVal);
    
    ch3[2] = V*0.2;
    merge(ch3, outputVal);
    Mat V_2 = convertHSV2BGR(outputVal);
    
    ch3[2] = V*0.4;
    merge(ch3, outputVal);
    Mat V_4 = convertHSV2BGR(outputVal);
    
    ch3[2] = V*0.6;
    merge(ch3, outputVal);
    Mat V_6 = convertHSV2BGR(outputVal);
    
    ch3[2] = V*0.8;
    merge(ch3, outputVal);
    Mat V_8 = convertHSV2BGR(outputVal);
    
    //Core 1
	// create a window for display and show our image inside it
	string img_display_core1 = "Core 1 Image Display";
	namedWindow(img_display_core1, WINDOW_AUTOSIZE);
	
    //row 1
    hconcat(B, G, B);
    hconcat(B, R, B);
    //row2
    hconcat(H, S, H);
    hconcat(H, V, H);
    //combine
    vconcat(B, H, B);
    imshow(img_display_core1, B);
   
   // save image
    imwrite("/Users/petadouglas/Documents/Uni/CGRA352/CGRA352_base/output/core1.png", B);

    //Core 2
    string img_display_core2 = "Core 2 Image Display";
    namedWindow(img_display_core2, WINDOW_AUTOSIZE);
    
    //row 1
    hconcat(H_0, H_2, H_0);
    hconcat(H_0, H_4, H_0);
    hconcat(H_0, H_6, H_0);
    hconcat(H_0, H_8, H_0);
    //row 2
    hconcat(S_0, S_2, S_0);
    hconcat(S_0, S_4, S_0);
    hconcat(S_0, S_6, S_0);
    hconcat(S_0, S_8, S_0);
    //combine rows 1 & 2
    vconcat(H_0, S_0, H_0);
    //row 3
    hconcat(V_0, V_2, V_0);
    hconcat(V_0, V_4, V_0);
    hconcat(V_0, V_6, V_0);
    hconcat(V_0, V_8, V_0);
    //combine
    vconcat(H_0, V_0, H_0);
    
    imshow(img_display_core2, H_0);

    // save image
    imwrite("/Users/petadouglas/Documents/Uni/CGRA352/CGRA352_base/output/core2.png", H_0);
    
    //Core 3
    string img_display_core3 = "Core 3 Image Display";
    namedWindow(img_display_core3, WINDOW_AUTOSIZE);
    
    Mat maskImg = mask(image);
    imshow(img_display_core3, maskImg);
    
    // save image
    imwrite("/Users/petadouglas/Documents/Uni/CGRA352/CGRA352_base/output/core3.png", maskImg);
    
    //Completion Laplacian
    string img_display_completion = "Completion Laplacian Image Display";
    namedWindow(img_display_completion, WINDOW_AUTOSIZE);
     
    Mat im = imread("/Users/petadouglas/Documents/Uni/CGRA352/CGRA352_base/work/res/Flower.jpg",0);
    Mat out = laplacian(im);
     
    imshow(img_display_completion, out);
     
    // save image
    imwrite("/Users/petadouglas/Documents/Uni/CGRA352/CGRA352_base/output/completion_laplacian.png", out);
    
    //Completion Sobel X direction
    string img_display_completion_sobx = "Completion Sobel xDir Image Display";
    namedWindow(img_display_completion_sobx, WINDOW_AUTOSIZE);
     
    Mat sobx_out = sobelX(im);
     
    imshow(img_display_completion_sobx, sobx_out);
     
    // save image
    imwrite("/Users/petadouglas/Documents/Uni/CGRA352/CGRA352_base/output/completion_sobel_x.png", sobx_out);
    
    //Completion Sobel Y direction
    string img_display_completion_soby = "Completion Sobel yDir Image Display";
    namedWindow(img_display_completion_soby, WINDOW_AUTOSIZE);
     
    Mat soby_out = sobelY(im);
     
    imshow(img_display_completion_soby, soby_out);
     
    // save image
    imwrite("/Users/petadouglas/Documents/Uni/CGRA352/CGRA352_base/output/completion_sobel_y.png", soby_out);
    
	// wait for a keystroke in the window before exiting
	waitKey(0);
    
 
}
