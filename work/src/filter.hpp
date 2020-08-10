//
//  filter.hpp
//  CGRA_PROJECT_cgra352
//
//  Created by Peta Douglas on 9/03/20.
//


#define filter_hpp

#include <stdio.h>
// opencv
#include <opencv2/core/core.hpp>

cv::Mat convertBGR2HSV(const cv::Mat &m);
cv::Mat convertHSV2BGR(const cv::Mat &m);
std::vector<cv::Mat> getChannels(const cv::Mat &m);
cv::Mat mask(const cv::Mat &m);
cv::Mat laplacian(const cv::Mat &m);
cv::Mat sobelX(const cv::Mat &m);
cv::Mat sobelY(const cv::Mat &m);
