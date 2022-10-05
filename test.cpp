#include "opencv2/opencv.hpp"
#include "math.h"
#include "stdio.h"
#include<opencv2/highgui/highgui.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <dirent.h>
#include "armadillo"
#include "opencv2/features2d.hpp"
#include "/home/karthik/opencv_build/opencv_contrib/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp"
#include "matplotlibcpp.h"
#include "opencv2/opencv.hpp"
#include <filesystem>
using namespace cv;
using namespace cv::xfeatures2d;
namespace plt = matplotlibcpp;

#ifndef __has_include
  static_assert(false, "__has_include not supported");
#else
#  if __cplusplus >= 201703L && __has_include(<filesystem>)
#    include <filesystem>
     namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
     namespace fs = std::experimental::filesystem;
#  elif __has_include(<boost/filesystem.hpp>)
#    include <boost/filesystem.hpp>
     namespace fs = boost::filesystem;
#  endif
#endif

cv::Mat stack(Mat &rotation1_fn, Mat &translation1_fn)
{
cv::Mat rot1_t,rot2_t,trans1_t,trans2_t;
rot1_t = rotation1_fn;
trans1_t = translation1_fn.t();

arma::mat rot1_arma_mat( reinterpret_cast<double*>(rot1_t.data), rot1_t.rows, rot1_t.cols );
arma::mat trans1_arma_mat( reinterpret_cast<double*>(trans1_t.data), trans1_t.rows, trans1_t.cols );
arma::mat proj_mat_i_arma_mat = join_vert(rot1_arma_mat,trans1_arma_mat);
return cv::Mat(proj_mat_i_arma_mat.n_cols,proj_mat_i_arma_mat.n_rows,CV_64F,proj_mat_i_arma_mat.memptr()).clone();


}

std::vector<Point_<float>> track_features(Mat &image1, Mat &image2, std::vector<Point_<float>> &p0)
{
    std::vector<Point_<float>> p1;
    goodFeaturesToTrack(image1, p0, 10000, 0.3, 7, Mat(), 7, false, 0.04);
    //fast feature detector
    Ptr<FastFeatureDetector> detector = cv::FastFeatureDetector::create(10,true);
    std::vector<KeyPoint> keypoints;
    detector->detect(image1, keypoints);
    Mat descriptors;
    detector->detect(image1, keypoints);
    cv::KeyPoint::convert(keypoints, p0);

    std::vector<uchar> status;
    std::vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(image1, image2,p0, p1, status, err, Size(15,15), 2, criteria);
          
    std::vector<Point2f> good_new,good_old;
    for(uint i = 0; i < p0.size(); i++)
    {
        // Select good points
        if(status[i] == 1) {
            good_new.push_back(p1[i]);
            good_old.push_back(p0[i]);
    }
    }
   
    p0 = good_old;
    return good_new;
}

//code to triangulate points

std::vector<Point3f> triangulate(Mat &calib,std::vector<Point2f> &p1, std::vector<Point2f> &p2, Mat &rotation1, Mat &translation1,Mat &rotation2, Mat &translation2)

{   Mat proj_mat_i, proj_mat_j;
    proj_mat_i= stack(rotation1, translation1);
    proj_mat_j= stack(rotation2, translation2);

    //stack rotation and translation matrices
    std::vector<Point3f> points;
    Mat points_4d;
    triangulatePoints(calib*proj_mat_i,calib*proj_mat_j, p1, p2, points_4d);
    for(uint i = 0; i < points_4d.cols; i++)
    {
        points.push_back(Point3f(points_4d.at<float>(0,i)/points_4d.at<float>(3,i),
                                 points_4d.at<float>(1,i)/points_4d.at<float>(3,i),
                                 points_4d.at<float>(2,i)/points_4d.at<float>(3,i)));
    }
    return points;
}



int main(int argc, char** argv)

    //read the images from the directory
{
    Mat my_image_array[200];
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir ("/home/karthik/vo_in_cpp/Visual-Odometry/KITTI_sample/images")) != NULL) {

      closedir (dir);
    } else {
      perror ("");
      return EXIT_FAILURE;
    }
    

    //sort images in the directory
    std::vector<std::string> filenames;
    std::string path = "/home/karthik/vo_in_cpp/Visual-Odometry/KITTI_sample/images";
    for (auto& p : std::filesystem::directory_iterator(path))
    {
        filenames.push_back(p.path().string());
    }
    std::sort(filenames.begin(), filenames.end());
    
    for (int i = 0; i < filenames.size(); i++)
    {
        my_image_array[i] = imread(filenames[i],IMREAD_GRAYSCALE);
    }
    
    Mat rotation1 = Mat::eye(3,3,CV_64F);
    Mat translation1 = Mat::zeros(3,1,CV_64F);

    double calib_array[3][3] = {{9.842439e+02,0.000000e+00,6.900000e+02},
    {0.000000e+00,9.808141e+02,2.331966e+02},
    {0.000000e+00,0.000000e+00,1.000000e+00}};
    Mat calib = Mat(3,3,CV_64F,calib_array);
   
    std::vector<KeyPoint> keypoints;
    std::vector<Point_<float>> p3,p2,p1;
    p2= track_features(my_image_array[0],my_image_array[1],p1);
    //find fundamental matrix
    Mat F = findFundamentalMat(p1,p2,FM_RANSAC,3,0.99);

    //finding transpose of the calibration matrix
    Mat calib_= calib.t();
        //multiply the transpose of the calibration matrix with the fundamental matrix
    Mat calib_F = calib_*F;
    //find the essential matrix
    Mat E = calib_F*calib;
    //find the rotation and translation vectors by recovering the essential matrix
    Mat rot, trans, mask;

    recoverPose(E, p1, p2, calib, rot, trans, mask);
    
    Mat rotation2 = rot;

    trans = rotation2*trans;
    Mat translation2= translation1-trans;

    std::vector<Mat> trans_dir,rot_dir;

    //code to print vector


    trans_dir.push_back(trans);
    rot_dir.push_back(rotation2);
    // p3=p1;
    std::vector<Point3f> new_cloud;
    new_cloud=triangulate(calib,p1,p2,rotation1,translation1,rot,trans);
    
    //doing this itertively

    std::vector<Mat>::iterator p,pRet;

    for(uint i = 1; i < 150; i++)
    {
        
        p2=track_features(my_image_array[i],my_image_array[i+1],p1);
        
        Mat F = findFundamentalMat(p1,p2,FM_RANSAC,3,0.99);
        Mat calib_= calib.t();
        Mat calib_F = calib_*F;
        //find the essential matrix
        Mat E = calib_F*calib;
        //find the rotation and translation vectors by recovering the essential matrix

        rot.release();
        trans.release();
        mask.release();
        recoverPose(E, p1, p2, calib, rot, trans, mask);
        rotation1 = rotation2;
        translation1 = translation2;
        std::vector<Point3f> old_cloud=new_cloud;
        rotation2=rotation1*rot;
        trans = rotation2*trans;
        new_cloud= triangulate(calib,p1,p2,rotation1,translation1,rot,trans);
        // double scale = relative(old_cloud,new_cloud);
        translation2 = translation1 - trans;

        translation1=translation2;
        rotation1 = rotation2;
        // std::cout<<translation2<<std::endl;
        p= trans_dir.end();
        pRet= trans_dir.insert(p,translation2.clone());
        p1=p2;


    }
std::vector<double> trans_dir_x,trans_dir_z;



   for(int i=0; i < trans_dir.size(); i++)
   std::cout << trans_dir.at(i) << ' ';
for(uint i = 1; i < 150; i++)
{
    Mat trans_dir_1=trans_dir[i];
    double trans_x = trans_dir_1.at<double>(0,0);
    double trans_z = trans_dir_1.at<double>(2,0);
    trans_dir_x.push_back(-1*trans_x);
    trans_dir_z.push_back(trans_z);
   

    // trans_dir_x.push_back(trans_dir[i][0][0])
	// trans_dir_z.push_back(trans_dir[i][2][0])
}

plt::figure();

//print trans_dir_z


std::cout<<"images loaded"<<std::endl;
plt::plot(trans_dir_x,trans_dir_z,"r--");
plt::savefig("minimal.pdf");
}


