/*
  *plan：
  2016/9/7
   1.reading all img 

  *problem：
    1.img compare by many img;
	2.modify programme format for standard c++
	3.strlen() arry pstr;
*/

#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv/ml.h"
#include <iostream>
#include "cv.h"
#include "highgui.h"
#include <vector>
#include <math.h>
#include <string.h>
#include <fstream>
#include<vector>
using namespace std;
using namespace cv;
//#pragma comment(lib,"opencv_highgui231d.lib")
const int re_size = 256; 
const int distanceThreshold = 100;
const int blur_size = 7;
const int nThresholdEdge = 25;
const int ratio = 3;
const int kernel_size = 3;
const float minError = 2.5;
/** 
     * @brief This function converts matches to points using nearest neighbor distance 
     * ratio matching strategy 
     * @param train Vector of keypoints from the first image 
     * @param query Vector of keypoints from the second image 
     * @param matches Vector of nearest neighbors for each keypoint 
     * @param pmatches Vector of putative matches 
     * @param nndr Nearest neighbor distance ratio value 
     */  
void matches2points_nndr(const vector<KeyPoint>& train,  
                             const vector<KeyPoint>& query,  
                             const vector<vector<DMatch> >& matches,  
                             vector<Point2f>& pmatches, float nndr) {  
      
      float dist1 = 0.0, dist2 = 0.0;  
      for (size_t i = 0; i < matches.size(); i++) {  
        DMatch dmatch = matches[i][0];  
        dist1 = matches[i][0].distance;  
        dist2 = matches[i][1].distance;  
      
        if (dist1 < nndr*dist2) {  
		  pmatches.push_back(train[dmatch.queryIdx].pt);  //说明在pmatches中第0个是的点是train的关键点即加载图片的关键点，第1个代表quary匹配到的关键点即数据库中的关键点
          pmatches.push_back(query[dmatch.trainIdx].pt);  
        }  
      }  
	  cout<<"number of match point"<<pmatches.size()/2<<endl;
    }
 /** 
     * @brief This function computes the set of inliers estimating the fundamental matrix 
     * or a planar homography in a RANSAC procedure 
     * @param matches Vector of putative matches 
     * @param inliers Vector of inliers 
     * @param error The minimum pixelic error to accept an inlier 
     * @param use_fund Set to true if you want to compute a fundamental matrix 
     */  
void compute_inliers_ransac(const std::vector<cv::Point2f>& matches,  
                                std::vector<cv::Point2f>& inliers,  
                                float error, bool use_fund) {  
      
      vector<Point2f> points1, points2;  
      Mat H = Mat::zeros(3,3,CV_32F);  //变换矩阵
      int npoints = matches.size()/2;  
      Mat status = Mat::zeros(npoints,1,CV_8UC1);  //标记矩阵
      
      for (size_t i = 0; i < matches.size(); i+=2) {  
        points1.push_back(matches[i]);  
        points2.push_back(matches[i+1]);  
      }  
      
      if (use_fund == true){  
        H = findFundamentalMat(points1,points2,CV_FM_RANSAC,error,0.99,status);  
      }  
      else {  
        H = findHomography(points1,points2,CV_RANSAC,error,status);  
      }  
      
      for (int i = 0; i < npoints; i++) {  
        if (status.at<unsigned char>(i) == 1) {    //status矩阵中，若为1 则表示该点为内点
          inliers.push_back(points1[i]);  
          inliers.push_back(points2[i]);  
        }  
      }  
	  cout<<"number of the inliers:"<<inliers.size()/2<<endl;
    }  
/**
  * @brief This function checks img edge by canny algorithm
  * @param mat roomImgCannySrc is source img 
  * @return return img through canny
*/
Mat cannyEdge(Mat roomImgCannySrc)
{
    blur(roomImgCannySrc, roomImgCannySrc, Size(blur_size,blur_size));
	equalizeHist(roomImgCannySrc,roomImgCannySrc);
	Mat roomImgCannnyDst;
	Canny(roomImgCannySrc, roomImgCannySrc, nThresholdEdge, nThresholdEdge * ratio, kernel_size); 
	imshow("roomImgCannnyDst",roomImgCannnyDst);
    return roomImgCannnyDst;
}
/*
   *@brief Thif function checks special point  
   *@param Mat roomImgSrc is sourc img 
   *@param vector roomImgKeyPoint is special point of source img
   *@return mat outRoomImgKeypoint which describle key point
*/
Mat specialPointCheck(Mat &roomImgSrc,vector<KeyPoint> &roomImgKeyPoint)
{
	SiftFeatureDetector  rommSiftDtc;
    rommSiftDtc.detect(roomImgSrc,roomImgKeyPoint);
	Mat *p = &roomImgSrc;
	Mat outRoomImgKeypoint;
	drawKeypoints(roomImgSrc,roomImgKeyPoint,outRoomImgKeypoint);
	return outRoomImgKeypoint;
}
int main()
{
	//static string imgPath = "G:\\graphic process\\roomGraphtic-v1\\roomGraphtic-v1\\G1.jpg";//读取源图
	static string imgPathG1 = "G9.jpg";//读取源图
	static string imgPathG2 = "G10.jpg";//读取源图
	Mat similarRoomImgSrc = imread(imgPathG1,CV_LOAD_IMAGE_COLOR);
	Mat similarRoomImgDst = imread(imgPathG2,CV_LOAD_IMAGE_COLOR);
	if(similarRoomImgSrc.empty()||similarRoomImgDst.empty())
	{
	  cout<<"cannot load roomImgSrc"<<endl;
	  return -1;
	}
	Mat roomImgResizeDstG1,roomImgResizeDstG2;
	resize(similarRoomImgSrc,roomImgResizeDstG1,Size(re_size,re_size),0,0,CV_INTER_LINEAR);
	resize(similarRoomImgDst,roomImgResizeDstG2,Size(re_size,re_size),0,0,CV_INTER_LINEAR);
	/*pyrDown(similarRoomImgSrc,roomImgDstG1,roomSzG1,BORDER_DEFAULT);//执行一次就减少原来的1/2
	pyrDown(similarRoomImgSrc,roomImgDstG1,roomSzG1,BORDER_DEFAULT);
	cvtColor(roomImgDstG1,roomImgDstG1,COLOR_BGR2GRAY);//转灰度图
	pyrDown(similarRoomImgDst,roomImgDstG2,roomSzG2,BORDER_DEFAULT);//执行一次就减少原来的1/2
	pyrDown(similarRoomImgDst,roomImgDstG2,roomSzG2,BORDER_DEFAULT);*/
	Mat roomImgColorDstG1,roomImgColorDstG2;
	cvtColor(roomImgResizeDstG1,roomImgColorDstG1,COLOR_BGR2GRAY);//转灰度图
	cvtColor(roomImgResizeDstG2,roomImgColorDstG2,COLOR_BGR2GRAY);//转灰度图
	//threshold(a1,a1,200,255,THRESH_BINARY);//二值化
	
	//图片G1 canny
	/*Mat roomImgCannnySrcG1 = roomImgColorDstG1.clone();
	Mat roomImgCannnySrcG2 = roomImgColorDstG2.clone();
	blur(roomImgCannnySrcG1, roomImgCannnySrcG1, Size(blur_size,blur_size));
	equalizeHist(roomImgCannnySrcG1,roomImgCannnySrcG1);
	Mat roomImgCannnyDstG1,roomImgCannnyDstG2;
	Canny(roomImgCannnySrcG1, roomImgCannnyDstG1, nThresholdEdge, nThresholdEdge * ratio, kernel_size); 
	imshow("roomImgCannnyDstG1",roomImgCannnyDstG1);*/
	//图片G1特征点检测
	//imshow("roomImgDstG1",roomImgColorDstG1);
	vector<KeyPoint> roomImgKeyPointG1;
	Mat drawImgKeyPointG1  = specialPointCheck(roomImgColorDstG1,roomImgKeyPointG1);
	imshow("drawImgKeyPointG1",drawImgKeyPointG1);
	cout<<"number of G1 special point:"<<roomImgKeyPointG1.size()<<endl;
	//图片G2特征点检测
	vector<KeyPoint> roomImgKeyPointG2;
	Mat drawImgKeyPointG2  = specialPointCheck(roomImgColorDstG2,roomImgKeyPointG2);
	imshow("drawImgKeyPointG2",drawImgKeyPointG2);
	cout<<"number of G2 special point:"<<roomImgKeyPointG2.size()<<endl;
	//提取特征向量
	SiftDescriptorExtractor siftExtractor;
	Mat descriptorG1,descriptorG2;
    siftExtractor.compute(roomImgColorDstG1,roomImgKeyPointG1,descriptorG1);
	siftExtractor.compute(roomImgColorDstG2,roomImgKeyPointG2,descriptorG2);
	//特征点匹配
    vector<vector<DMatch>> matches;
    Mat img_matches;
	imshow("desc",descriptorG1);
	BruteForceMatcher<L2<float>> matcher;
    matcher.knnMatch(descriptorG1,descriptorG2,matches,2);//寻找每个查询特征关键点对应的``k``个最佳匹配.
	cout<<"the count:"<<matches.size()<<endl;
    drawMatches(roomImgColorDstG1,roomImgKeyPointG1,roomImgColorDstG2,roomImgKeyPointG2,matches,img_matches);//画出匹配图
    imshow("matches",img_matches);
	//相似判断
	vector<Point2f> matchesSift, inlierSift;  
	float matchRatio = 0.8;
	matches2points_nndr(roomImgKeyPointG1,roomImgKeyPointG2,matches,matchesSift,matchRatio);
	compute_inliers_ransac(matchesSift,inlierSift,minError,false);
	waitKey();
	return 0;
}
