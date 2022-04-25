
#include <iostream>
#include <fstream>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/optflow.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

std::string root_path = "/Users/payne/Desktop/project/code/slam_practice/data";


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
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr DepthToPointCloud(const cv::Mat &depth, const cv::Mat rgb, const cv::Mat camera_matrix, float sensor_scale) {
   // 
   float cx = camera_matrix.at<float>(0,2);
   float cy = camera_matrix.at<float>(1,2);
   float fx = camera_matrix.at<float>(0,0);
   float fy = camera_matrix.at<float>(1,1);

   using cloud_type = pcl::PointCloud<pcl::PointXYZRGBA>;
   cloud_type::Ptr point_cloud (new cloud_type);
   for(auto y = 0; y < depth.rows; ++y) {
       const ushort *pd = depth.ptr<ushort>(y);
       const cv::Vec3b* pc = rgb.ptr<cv::Vec3b>(y);
       for(auto x = 0; x < depth.cols; ++x) {
           float zw = pd[x] * sensor_scale;
           float xw = (x - cx) * zw/fx;
           float yw = (y - cy) * zw/fy;

           cloud_type::PointType pt;
           pt.x = xw;
           pt.y = yw;
           pt.z = zw;
           pt.r = pc[x][0];
           pt.g = pc[x][1];
           pt.b = pc[x][2];
          
          point_cloud->push_back(pt);
           
       }
   }
   point_cloud->height = 1;
   point_cloud->width = point_cloud->size();
   return point_cloud;
}
const double camera_factor = 1000;
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;

struct FRAME{
    cv::Mat img_rgb;
    cv::Mat img_depth;
    std::vector<cv::KeyPoint> img_kpt;
    cv::Mat img_kpt_desc;
};

void FeaturesDection(const cv::Mat img, cv::Mat *img_kpt_desc, std::vector<cv::KeyPoint> *img_kpt) {
        // feature detection
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat keypoints_desc;
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create(0, 3, 0.01, 100, 1.6, CV_32F);
        detector->detectAndCompute(img, cv::Mat(), keypoints, keypoints_desc);
        std::cout<<"-- feature detect done. keypoint size "<<keypoints.size()<<std::endl;
        *img_kpt_desc = keypoints_desc.clone();
        img_kpt->swap(keypoints);
}

void FeaturesMatching(std::vector<cv::KeyPoint> img_kpt1, const cv::Mat img_kpt_desc1,
                      std::vector<cv::KeyPoint> img_kpt2, const cv::Mat img_kpt_desc2,
                      std::vector<cv::Point2f> *img_kpt1_match, std::vector<cv::Point2f> *img_kpt2_match) {
        // feature match
        cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce" );
        

        std::vector<std::vector<cv::DMatch>> matches;
        matcher->knnMatch(img_kpt_desc1, img_kpt_desc2, matches, 2);

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

        // 
        for ( int i = 0; i < matches.size(); i++)
        {
            if (matches[i][0].distance >0.6 * matches[i][1].distance) continue;
            if(matches[i][0].distance <= 5 * min_dist)
            {
                good_matches.push_back(matches[i][0]);
                img_kpt1_match->push_back(img_kpt1[matches[i][0].queryIdx].pt);
                img_kpt2_match->push_back(img_kpt2[matches[i][0].trainIdx].pt);
            }
        }
}
void LoadData(int img_id, FRAME *frame){
        cv::Mat depth = cv::imread(root_path +"/depth_png/"+std::to_string(img_id)+".png", -1);
        cv::Mat rgb = cv::imread(root_path + "/rgb_png/"+std::to_string(img_id)+".png", 1);
        frame->img_rgb = rgb.clone();
        frame->img_depth = depth.clone();
        std::cout<<"-- load rgb data "<<depth.size()<<" type "<<depth.type()<<" "<<rgb.size()<<std::endl;
}
int main(int argc, char* argv[]) {
    
    using cloud_type = pcl::PointCloud<pcl::PointXYZRGBA>;
    cv::Mat camera_matrix = (cv::Mat_<float>(3,3)<<camera_fx,0, camera_cx, 0, camera_fy, camera_cy, 0, 0, 1);

    // show
    // pcl::visualization::CloudViewer* cloud_viewer(new pcl::visualization::CloudViewer("viewer"));
    FRAME frame_cur, frame_last;
    std::vector<cv::KeyPoint> last_kps;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr last_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
    for(auto img_id = 1; img_id < 3; ++img_id) {


        LoadData(img_id, &frame_cur);
        cloud_type::Ptr point_cloud = DepthToPointCloud(frame_cur.img_depth, frame_cur.img_rgb, camera_matrix, camera_factor);
        cv::Mat img_kpt_desc;
        std::vector<cv::KeyPoint> img_kpt;
        FeaturesDection(frame_cur.img_rgb, &img_kpt_desc, &img_kpt);

        // feature match
        if (img_id == 1) {
            frame_last = frame_cur;
            frame_last.img_kpt_desc = img_kpt_desc.clone();
            frame_last.img_kpt.swap(img_kpt);

           *last_point_cloud = *point_cloud;
        } else {

        // feature match

        std::vector<cv::Point2f> keypoints_curr, keypoints_last;
        FeaturesMatching(img_kpt, img_kpt_desc, frame_last.img_kpt, frame_last.img_kpt_desc, &keypoints_curr, &keypoints_last);

        // keypoint 3d to 2d
        std::vector<cv::Point3f> last_kps_3d;
        for(auto pt : keypoints_last) {
            float zw = frame_last.img_depth.ptr<ushort>(int(pt.y))[int(pt.x)]/ camera_factor;
            float xw = (pt.x - camera_cx) * zw/camera_fx;
            float yw = (pt.y - camera_cy) * zw/camera_fy;
            last_kps_3d.push_back(cv::Point3f(xw, yw, zw));
        }

        std::cout<<"-- Pose Estimate "<<std::endl;
        // pose estimate
        cv::Mat R, T;
        int ret = cv::solvePnPRansac(last_kps_3d, keypoints_curr, camera_matrix, cv::Mat(), R, T);
        cv::Mat r;
        cv::Rodrigues(R, r);
        Eigen::Matrix4d transform_1 = Eigen::Matrix4d::Identity();
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                transform_1(i,j) = r.at<double>(i,j);
            }
        }
        transform_1(0,3)=T.at<double>(0,0);
        transform_1(1,3)=T.at<double>(1,0);
        transform_1(2,3)=T.at<double>(2,0);
        // std::cout<<"-- solvePnPRansac rvec, tvec "<<transform_1<<std::endl;
        std::cout<<"-- 3d-2d estimate "<<" "<<r<<" "<<T<<std::endl;       

        // 2d - 2d
        cv::Mat mask, R2d, T2d;
        cv::Mat E = cv::findEssentialMat(keypoints_curr, keypoints_last, camera_matrix, cv::RANSAC, 0.999, 1.0, 1000, mask);
        cv::recoverPose(E, keypoints_curr, keypoints_last, camera_matrix, R2d, T2d);
        std::cout<<"-- 2d-2d estimate "<<R2d<<" "<<T2d<<std::endl;

       // 3d - 3d
        pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
        icp.setInputSource(last_point_cloud);
        icp.setInputTarget(point_cloud);
        
        pcl::PointCloud<pcl::PointXYZRGBA> Final;
        icp.align(Final);
      
        std::cout << "has converged:" << icp.hasConverged() << " score: " <<
        icp.getFitnessScore() << std::endl;
        std::cout << icp.getFinalTransformation() << std::endl;


        // point cloud transform
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>());
       
         // Create the filtering object
         pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
         sor.setInputCloud (point_cloud);
         sor.setLeafSize (0.3f, 0.3f, 0.3f);
         sor.filter (*cloud_filtered);
         *point_cloud = *cloud_filtered;

        //
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr new_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
        pcl::transformPointCloud(*last_point_cloud, *new_point_cloud, transform_1);
        *point_cloud += *new_point_cloud;
        *last_point_cloud = *point_cloud;
        
        frame_last.img_kpt_desc = img_kpt_desc.clone();
        frame_last.img_kpt.swap(img_kpt);

        // cloud_viewer->showCloud(last_point_cloud);

        }
        
    
    }

        // pcl::io::savePCDFile("point_cloud.pcd", *last_point_cloud);


    // show

    // pnp


    return 1;
/*

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
    */
}
