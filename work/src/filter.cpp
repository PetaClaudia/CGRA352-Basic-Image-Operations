//
//  filter.cpp
//  CGRA_PROJECT_cgra352
//
//  Created by Peta Douglas on 9/03/20.
//

#include "filter.hpp"
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
using namespace std;
using namespace cv;

Mat convertBGR2HSV(const cv::Mat &m){
    Mat im = m.clone();
    Mat hsvImg;
    cvtColor(im, hsvImg, CV_BGR2HSV);
    return hsvImg;
}

Mat convertHSV2BGR(const cv::Mat &m){
    Mat im = m.clone();
    Mat bgrImg;
    cvtColor(im, bgrImg, CV_HSV2BGR);
    return bgrImg;
}

vector<cv::Mat> getChannels(const cv::Mat &m){
    Mat im = m.clone();
    std::vector<cv::Mat> channels;
    split(im, channels);
    return channels;
}

Mat mask(const cv::Mat &m){
    Mat im = m.clone();
    Mat maskImg;
    unsigned char * point = im.ptr(80,80);
    int b1 = point[0];
    int g1 = point[1];
    int r1 = point[2];
    
    //go through every pixel in image.
    for(int i = 0; i < im.rows; i++){
        for (int j = 0; j< im.cols; j++){
            Vec3b pt = im.at<cv::Vec3b>(i,j);
            int b2 = pt[0];
            int g2 = pt[1];
            int r2 = pt[2];
            int thresh = sqrt(((r1-r2)*(r1-r2))+((g1-g2)*(g1-g2))+((b1-b2)*(b1-b2)));
            
            if(thresh<100){
                //set to white
                im.at<cv::Vec3b>(i,j)[0] = 255;
                im.at<cv::Vec3b>(i,j)[1] = 255;
                im.at<cv::Vec3b>(i,j)[2] = 255;
            }
            else{
                //set to black
                im.at<cv::Vec3b>(i,j)[0] = 0;
                im.at<cv::Vec3b>(i,j)[1] = 0;
                im.at<cv::Vec3b>(i,j)[2] = 0;
            }
        }
    }
    return im;
}

Mat laplacian(const cv::Mat &m){
    Mat im = m.clone();
    int s = 1; // uniform spacing
    int minVal, maxVal;
    int range, norm;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            
            int sum = (4 * m.at<uchar>(i,j)
                       - m.at<uchar>(i-s,j-s)
                       - m.at<uchar>(i-s,j+s)
                       - m.at<uchar>(i+s,j-s)
                       -m.at<uchar>(i+s,j+s))/s*s;
            
            if(sum > maxVal){
                maxVal = sum;
            }
            if(sum < minVal){
                minVal = sum;
            }
            
        }
    }
    cout << "lap: minVal : " << minVal << endl << "maxVal : " << maxVal << endl;
    range = abs(minVal)+abs(maxVal);
    norm = range/255;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            
            int sum = (4 * m.at<uchar>(i,j)
                       - m.at<uchar>(i-s,j-s)
                       - m.at<uchar>(i-s,j+s)
                       - m.at<uchar>(i+s,j-s)
                       -m.at<uchar>(i+s,j+s))/s*s;
            
            im.at<uchar>(i, j) = ((sum-minVal)/-norm)+0.5;
            cout << "val : " << (sum-minVal)/-norm<< endl;
        }
    }
    return im;
}

Mat sobelX(const cv::Mat &m){
    Mat im = m.clone();
    int gx, sum, minVal = 0, maxVal = 0;
    double range, norm;
    
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            
            //find x gradient
            sum = m.at<uchar>(i-1, j-1)
            + 2*m.at<uchar>(i, j-1)
            + m.at<uchar>(i+1, j-1)
            - m.at<uchar>(i-1, j+1)
            - 2*m.at<uchar>(i, j+1)
            - m.at<uchar>(i+1, j+1);
            
            if(sum > maxVal){
                maxVal = sum;
            }
            if(sum < minVal){
                minVal = sum;
            }
        }
    }
    cout << "gx: minVal : " << minVal << endl << "maxVal : " << maxVal << endl;
    range = (abs(minVal)+abs(maxVal));
    norm = range/255;
    
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            
            //find x gradient
            gx = m.at<uchar>(i-1, j-1)
            + 2*m.at<uchar>(i, j-1)
            + m.at<uchar>(i+1, j-1)
            - m.at<uchar>(i-1, j+1)
            - 2*m.at<uchar>(i, j+1)
            - m.at<uchar>(i+1, j+1);
            
            im.at<uchar>(i, j) = ((gx-minVal)/-norm)+0.5;
        }
    }
    return im;
}

Mat sobelY(const cv::Mat &m){
    Mat im = m.clone();
    int gy, sum, minVal = 0, maxVal = 0;
    double range, norm;
    
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            sum = m.at<uchar>(i-1, j-1)
            + 2*m.at<uchar>(i-1, j)
            + m.at<uchar>(i-1, j+1)
            - m.at<uchar>(i+1, j-1)
            - 2*m.at<uchar>(i+1, j)
            - m.at<uchar>(i+1, j+1);
           // sum = sqrt(gy*gy);
            if(sum > maxVal){
                maxVal = sum;
            }
            if(sum < minVal){
                minVal = sum;
            }
        }
    }
    cout << "gy: minVal : " << minVal << endl << "maxVal : " << maxVal << endl;
    range = abs(minVal)+abs(maxVal);
    norm = range/255;
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            
            //find y gradient
            gy = m.at<uchar>(i-1, j-1)
                + 2*m.at<uchar>(i-1, j)
                + m.at<uchar>(i-1, j+1)
                - m.at<uchar>(i+1, j-1)
                - 2*m.at<uchar>(i+1, j)
                - m.at<uchar>(i+1, j+1);
            //sum = sqrt(gy*gy);
            //norm = max(abs(minVal), (abs(maxVal)));
            //im.at<uchar>(i, j) = sum;
            //im.at<uchar>(i, j) = 255*((im.at<uchar>(i, j)/2*norm)+0.5);
            im.at<uchar>(i, j) = ((gy-minVal)/-norm)+0.5;
        }
    }
    return im;
}
