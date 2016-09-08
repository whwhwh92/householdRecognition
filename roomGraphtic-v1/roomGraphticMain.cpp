/*
  *plan：
  2016/9/7
   1.reading all img 
  2016/9/8
   1.search img in all img 
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
#include <io.h>
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
bool read_all_img_name(vector<string> &imgPathName,string &fileNameJpg,string & fileName)
{
   _finddata_t fileInfo;
    int num = 0;
	long handle = _findfirst(fileNameJpg.c_str(), &fileInfo);
	if (handle == -1L)
    {
        cout << "failed to transfer files" << endl;
        return false;
    }

    do 
    {
        num ++;
        //cout << fileInfo.name <<endl;
		string imgname = fileInfo.name;
		string img_path_name = fileName+"\\"+imgname ;
		imgPathName.push_back(img_path_name);
    } while (0 == _findnext(handle, &fileInfo) );
    cout << " .exe files' number:  " << num << endl;
	for(int i =0;i<imgPathName.size();i++)
	{
	   cout<<imgPathName[i]<<endl;
	}
	return true;
}
void img_preprocess(Mat &queryImg,vector<Mat> &imgSrc)
{
   resize(queryImg,queryImg,Size(re_size,re_size),0,0,CV_INTER_LINEAR);
   cvtColor(queryImg,queryImg,COLOR_BGR2GRAY);
   for(int num = 0;num < imgSrc.size();num++)
   {
     resize(imgSrc[num],imgSrc[num],Size(re_size,re_size),0,0,CV_INTER_LINEAR);
    cvtColor(imgSrc[num],imgSrc[num],COLOR_BGR2GRAY);
   }
}
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
int compute_inliers_ransac(const std::vector<cv::Point2f>& matches,  
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
		  if(matches.size()>6)  //避免matches.size()/2>3的情况 这样会到时程序崩溃
		  {
		     H = findHomography(points1,points2,CV_RANSAC,error,status); 
		  }
		  else
		  {
		     cout<<"matches size less than 3"<<endl;
			 return -1;
		  }
      }  
      
      for (int i = 0; i < npoints; i++) {  
        if (status.at<unsigned char>(i) == 1) {    //status矩阵中，若为1 则表示该点为内点
          inliers.push_back(points1[i]);  
          inliers.push_back(points2[i]);  
        }  
      }  
	  cout<<"number of the inliers:"<<inliers.size()/2<<endl;
	  return 0;
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
vector<KeyPoint>  sift_special_point(Mat &siftImgSrc)
{
	vector<KeyPoint> roomImgKeyPoint;
    SiftFeatureDetector  SiftDtc;
    SiftDtc.detect(siftImgSrc,roomImgKeyPoint);
	return roomImgKeyPoint;
}
/*
 *@brief this function extractor descriptor on keypoint of img
 *@param mat siftImgSrc is source img which need extractor descriptor
 *@param keypoint roomImgKeyPoint from special point of img
 *return descriptor of img
*/
Mat sift_descriptor_extractor(Mat &siftImgSrc,vector<KeyPoint> &roomImgKeyPoint)
{
	Mat descriptor;
	SiftDescriptorExtractor siftExtractor;
	
    siftExtractor.compute(siftImgSrc,roomImgKeyPoint,descriptor);
	return descriptor;
	
}
/*
 *@brief this function match special  Descriptor of img
 *@param vector<vector<DMatch>> Matches contian match index and distance of two key point
 *vector<vector<vector<DMatch>>> allMatches contian matches of all img in file
 *@param queryDescriptor is Descriptor of query img (load img)
 *@param srcDescriptor is Descriptor of query img in file
*/

void special_descriptor_match(vector<vector<vector<DMatch>>> &allMatches,Mat &queryDescriptor,vector<Mat> &srcDescriptor)
{
    Mat img_matches;
	BruteForceMatcher<L2<float>> matcher;
	for(int num = 0;num < srcDescriptor.size();num++)
	{
	 vector<vector<DMatch>> matches;
	  matcher.knnMatch(queryDescriptor,srcDescriptor[num], matches,2);//寻找每个查询特征关键点对应的``k``个最佳匹配.
	  allMatches.push_back(matches);
	}
}
/*
  *@brief this function find max num of two img,in addition ,getted serial of max num
  * finded file path of match img by serial
  *@param vector<vector<Point2f>> match contain all match point of file and query
  *@param serialMaxData serial is serial of the biggest match num 
*/
void find_max_match_num(vector<vector<Point2f>> &match ,int &serialMaxData)
{
	int maxData = match[0].size();
	
	for(int num = 0;num <match.size();num++)
	{
		if(match[num].size() > maxData )
		{
			maxData = match[num].size();
			serialMaxData = num ;
		}
	}
}
int main()
{
	static string imgPathJpg = "G:\\graphicprocess\\roomGraphtic-v1\\photo\\*.jpg";//读取源图
	static string imgPath = "G:\\graphicprocess\\roomGraphtic-v1\\photo";
	//static string queryImgPath;
	vector<Mat> roomImgSrc ;
	vector<string> imgPathName;
	read_all_img_name(imgPathName,imgPathJpg,imgPath);
	for(int num = 0;num<imgPathName.size();num++)
	{
	   Mat imgSrc = imread(imgPathName[num],CV_LOAD_IMAGE_COLOR);
	   roomImgSrc.push_back(imgSrc);
	   if(imgSrc.empty())
	   {
	      cout<<"failed src load :"<<imgPathName[num]<<endl;
		  return -1;
	   }
	}
	Mat queryImg = imread("queryImg.jpg",CV_LOAD_IMAGE_COLOR);
	if(queryImg.empty())
	{
	  cout<<"failed query load"<<endl;
	  return -1;
	}
	img_preprocess(queryImg,roomImgSrc);
	imshow("queryImg",queryImg);
	/*pyrDown(similarRoomImgSrc,roomImgDstG1,roomSzG1,BORDER_DEFAULT);//执行一次就减少原来的1/2
	pyrDown(similarRoomImgSrc,roomImgDstG1,roomSzG1,BORDER_DEFAULT);
	cvtColor(roomImgDstG1,roomImgDstG1,COLOR_BGR2GRAY);//转灰度图
	pyrDown(similarRoomImgDst,roomImgDstG2,roomSzG2,BORDER_DEFAULT);//执行一次就减少原来的1/2
	pyrDown(similarRoomImgDst,roomImgDstG2,roomSzG2,BORDER_DEFAULT);*/
	//图片G1 canny
	/*Mat roomImgCannnySrcG1 = roomImgColorDstG1.clone();
	Mat roomImgCannnySrcG2 = roomImgColorDstG2.clone();
	blur(roomImgCannnySrcG1, roomImgCannnySrcG1, Size(blur_size,blur_size));
	equalizeHist(roomImgCannnySrcG1,roomImgCannnySrcG1);
	Mat roomImgCannnyDstG1,roomImgCannnyDstG2;
	Canny(roomImgCannnySrcG1, roomImgCannnyDstG1, nThresholdEdge, nThresholdEdge * ratio, kernel_size); 
	imshow("roomImgCannnyDstG1",roomImgCannnyDstG1);*/
	//图片queryImg特征点检测
	//imshow("roomImgDstG1",roomImgColorDstG1);
	vector<KeyPoint> queryImgKeyPoint;
	Mat queryDescriptor;
	queryImgKeyPoint = sift_special_point(queryImg);
	queryDescriptor = sift_descriptor_extractor(queryImg,queryImgKeyPoint);
	//图片src特征点检测
	vector<KeyPoint> roomImgKeyPoint;
	vector<vector<KeyPoint>> roomImgAllKeyPoint;
	vector<Mat> srcDescriptor;

	for(int num = 0;num<roomImgSrc.size();num++)
	{
	  vector<KeyPoint> roomImgAllKeyTempPoint = sift_special_point(roomImgSrc[num]);
	  roomImgAllKeyPoint.push_back(roomImgAllKeyTempPoint);
	  Mat srcDescriptorTemp = sift_descriptor_extractor(roomImgSrc[num],roomImgAllKeyTempPoint);
	  srcDescriptor.push_back(srcDescriptorTemp);
	}
	
	//特征向量匹配
    vector<vector<vector<DMatch>>> allMatches;
    Mat img_matches;
	BruteForceMatcher<L2<float>> matcher;
    special_descriptor_match(allMatches,queryDescriptor,srcDescriptor);
   
	//取出匹配点
	vector<vector<Point2f>> matchesSift, inlierSift;  
	float matchRatio = 0.8;
	for(int num = 0;num < roomImgSrc.size();num++)
	{
	  vector<Point2f> matchesSiftTemp,inlierSiftTemp;
	  matches2points_nndr(queryImgKeyPoint,roomImgAllKeyPoint[num],allMatches[num],matchesSiftTemp,matchRatio);
	  matchesSift.push_back(matchesSiftTemp);
	  compute_inliers_ransac(matchesSift[num],inlierSiftTemp,minError,false);
	  inlierSift.push_back(inlierSiftTemp);
	}

	//找到匹配的图片
	int serialMaxVectorSize = 0;
	find_max_match_num(inlierSift,serialMaxVectorSize);
	string matchStringPath = imgPathName[serialMaxVectorSize];
	Mat matchImg  = imread(matchStringPath,CV_LOAD_IMAGE_COLOR);
	imshow("matchImg",matchImg);
	waitKey();
	return 0;
}
