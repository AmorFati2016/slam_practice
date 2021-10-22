
#include <iostream>
#include <fstream>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/optflow.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

struct Frame{
// data
    std::vector<cv::KeyPoint> vkey_pts_;
    cv::Mat descriptors_;
    cv::Mat rotate_;
    cv::Mat trans_;
    std::vector<cv::Point3f> vkey_pts_3d_visible_;
    std::vector<int> vkey_pts_3d_visible_idx_;
};

std::vector<Frame> vframe;
cv::Mat last_image;


int main(int argc, char* argv[]) {

    // 1. load image names
    std::vector<std::string> left_img_path, right_img_path;
    cv::Mat left_img, right_img;
    std::string left_img_root = argv[1];
    std::string right_img_root = argv[2];
    std::string times_stamp_path = argv[3];
    std::cout<<"-- left_img_root "<<left_img_root<<" right_img_root "<<right_img_root<<" times_stamp_path "<<times_stamp_path<<std::endl;
    std::ifstream io_timestamp;
    io_timestamp.open(times_stamp_path);
    while(!io_timestamp.eof()) {
        std::string str;
        getline(io_timestamp, str);

        if (!str.empty()) {
        std::stringstream ss;
        ss << str;

        left_img_path.push_back(left_img_root + "/" + ss.str() + ".jpg");
        right_img_path.push_back(right_img_root + "/" + ss.str() + ".jpg");
        }
    }

    // 2. load camera params    
    float f = 1.2 * std::max(left_img.cols, left_img.rows);
    float cx = left_img.cols / 2;
    float cy = left_img.rows / 2;
    cv::Mat cameraMatrix = (cv::Mat1f(3,3)<<f, 0, cx, 0, f, cy, 0, 0, 1);

    int num_imgs = left_img_path.size();
    for(int img_id = 0; img_id < num_imgs; ++img_id) {

    std::cout<<"-- Load data ... "<<left_img_path[img_id]<<std::endl;
    // frame

    Frame frame;

    // data load
    left_img = cv::imread(left_img_path[img_id], 0);


    float f = 1.2 * std::max(left_img.cols, left_img.rows);
    float cx = left_img.cols / 2;
    float cy = left_img.rows / 2;
    cameraMatrix.at<float>(0, 0) = f; cameraMatrix.at<float>(1, 1) = f;
    cameraMatrix.at<float>(0, 2) = cx; cameraMatrix.at<float>(1, 2) = cy;

    
    // feature match
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat keypoints_desc;
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(0, 3, 0.01, 100, 1.6, CV_32F);
    detector->detectAndCompute(left_img, cv::Mat(), keypoints, keypoints_desc);
    std::cout<<"-- feature detect done. keypoint size "<<keypoints.size()<<std::endl;


    frame.vkey_pts_ = keypoints;
    frame.descriptors_ = keypoints_desc.clone();
    
    // feature match
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce" );
    std::vector<cv::Point2f> keypoints_curr, keypoints_last;

    if (img_id == 0) {

    } else {
        std::vector<std::vector<cv::DMatch>> matches;
        matcher->knnMatch(keypoints_desc, vframe[img_id - 1].descriptors_, matches, 2);

        double min_dist = 10000.0, max_dist = 0.0;
        
        for ( int i = 0; i < matches.size(); i++ )
        {
            if (matches[i][0].distance >0.6 * matches[i][1].distance) continue;
            double dist = matches[i][0].distance;
            if ( dist < min_dist ) min_dist = dist;
            if ( dist > max_dist ) max_dist = dist;
        }

        std::cout<<"-- matches "<<matches.size()<<" min_dist "<<min_dist<<" max_dist "<<max_dist<<std::endl;
        std::vector<cv::DMatch> good_matches;

        for ( int i = 0; i < matches.size(); i++)
        {
            if (matches[i][0].distance >0.6 * matches[i][1].distance) continue;
            if(matches[i][0].distance <= 5 * min_dist)
            {
                good_matches.push_back(matches[i][0]);
                keypoints_curr.push_back(keypoints[matches[i][0].queryIdx].pt);
                keypoints_last.push_back(vframe[img_id - 1].vkey_pts_[matches[i][0].trainIdx].pt);
            }
        }
        
    }
    

    if (img_id == 0) {
        frame.rotate_ = cv::Mat::eye(3,3,CV_32FC1);
        frame.trans_ = cv::Mat::zeros(3,1, CV_32FC1);
        vframe.push_back(frame);
        last_image = left_img.clone();
        continue;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud ( new pcl::PointCloud<pcl::PointXYZ>);
            


    // compute 3d coordinate
     if (img_id == 1) {


        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(keypoints_last, keypoints_curr, cameraMatrix, 8, 0.999, 1.0, mask);

        cv::Mat R, T;
        cv::recoverPose(E, keypoints_last, keypoints_curr, cameraMatrix, R, T, mask);
        std::cout<<"-- E "<<E
                 <<"-- R "<<R
                 <<"-- T "<<T<<std::endl;

        R.convertTo(R, CV_32FC1);
        T.convertTo(T, CV_32FC1);

        std::cout<<"-- R "<<R<<" T "<<T<<std::endl;

        frame.rotate_ = R.clone();
        frame.trans_ = T.clone();

        // Create two relative pose
        // P1 = K [  I    |   0  ]
        // P2 = K [R{1,2} | {+-}t]
        cv::Mat proj_ref = cv::Mat::zeros(3,4, CV_32FC1);
        cv::Mat proj_cur = cv::Mat::zeros(3,4, CV_32FC1);


        cv::Mat m1 = vframe[img_id - 1].rotate_;
        cv::Mat t1 = vframe[img_id - 1].trans_;
        cv::hconcat(cameraMatrix * m1, t1, proj_ref);
        cv::hconcat(cameraMatrix * R, cameraMatrix * T, proj_cur);

        cv::Mat proj_pts;
        cv::triangulatePoints(proj_ref, proj_cur, keypoints_last, keypoints_curr, proj_pts);

        // normalize 3d pts
        for(int y = 0; y < proj_pts.cols; ++y) {
            pcl::PointXYZ p;
            p.x = proj_pts.at<float>(0,i) / proj_pts.at<float>(3,i);
            p.y = proj_pts.at<float>(1,i) / proj_pts.at<float>(3,i);
            p.z = proj_pts.at<float>(2,i) / proj_pts.at<float>(3,i);
            cloud->points.push_back(p);
        }

     } else if (img_id > 1) {

     }
    
    vframe.push_back(frame);
    std::cout<<"-- new coordinate "<<pt3<<std::endl;


    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;
    pcl::io::savePCDFile( "./pointcloud_"+std::to_string(img_id)+".pcd", *cloud );

    cv::waitKey(30);

    }

    // 3. end
    // std::cout<<"-- shutdown"<<std::endl;
}
