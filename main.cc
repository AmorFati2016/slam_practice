
#include <iostream>
#include <fstream>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

struct Frame{
// data
    std::vector<cv::KeyPoint> vkey_pts_;
    cv::Mat descriptors_;
    cv::Mat rotate_;
    cv::Mat trans_;
    std::vector<cv::Point3f> vkey_pts_3d_;
    std::vector<int> vkey_pts_visible_with_last_frame_;
};

std::vector<Frame> vframe;


int main(int argc, char* argv[]) {
    std::cout<<" enter slam system "<<std::endl;
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

        left_img_path.push_back(left_img_root + "/" + ss.str());
        right_img_path.push_back(right_img_root + "/" + ss.str());
        }
    }

    // 2. load camera params
    // cv::Mat cam0 = cv::Mat::

    cv::Mat distCoeffs = (cv::Mat1f(4,1)<<0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182);
    cv::Mat cameraMatrix = (cv::Mat1f(3,3)<<190.97847715128717, 0, 254.93170605935475, 0, 190.9733070521226, 256.8974428996504, 0, 0, 1);


    std::cout<<"-- Load data ... ";
    cv::Mat screen = cv::Mat::zeros(1000, 1000, CV_8UC1);
    cv::Point3f pt3 = cv::Point3f(500,500,0);
    int num_imgs = left_img_path.size();
    for(int img_id = 0; img_id < num_imgs; ++img_id) {

    std::cout<<"-- Load data ... "<<left_img_path[img_id]<<std::endl;
    // frame

    Frame frame;

    // data load
    left_img = cv::imread(left_img_path[img_id], 0);
    right_img = cv::imread(right_img_path[img_id], 0);

    cv::Mat left, right;
    cv::fisheye::undistortImage(left_img, left_img, cameraMatrix, distCoeffs, cameraMatrix, left_img.size());
    // cv::fisheye::undistortImage(right_img, right_img, cameraMatrix, distCoeffs, cameraMatrix, right_img.size());




    // feature detect
    cv::Ptr<cv::ORB> detector = cv::ORB::create(1000);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat keypoints_desc;
    detector->detectAndCompute(left_img, cv::Mat(), keypoints, keypoints_desc);
    frame.vkey_pts_ = keypoints;
    frame.descriptors_ = keypoints_desc.clone();
    frame.vkey_pts_visible_with_last_frame_.resize(keypoints.size(), 0);
    frame.vkey_pts_3d_.resize(keypoints.size(), cv::Point3f(0,0,0));
 


     // 绘制关键点
    cv::Mat keypoint_img;
    cv::drawKeypoints(left_img, keypoints, keypoint_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    
    // feature match
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
    std::vector<cv::Point2f> keypoints_curr, keypoints_last;

    std::vector<int> cur_pts_id_with_last_last, last_pts_id_com_with_last_last;

    if (img_id == 0) {

    } else {
        std::vector<cv::DMatch> matches;
        matcher->match(keypoints_desc, vframe[img_id - 1].descriptors_, matches);

        double min_dist = 10000.0, max_dist = 0.0;
        for ( int i = 0; i < keypoints_desc.rows; i++ )
        {
            double dist = matches[i].distance;
            if ( dist < min_dist ) min_dist = dist;
            if ( dist > max_dist ) max_dist = dist;
        }

        std::cout<<"-- matches "<<matches.size()<<std::endl;
        std::vector<cv::DMatch> good_matches;
        for ( int i = 0; i < keypoints_desc.rows; i++)
        {
            if(matches[i].distance <= std::max(2 * min_dist, 130.0))
            {
                good_matches.push_back(matches[i]);
                keypoints_curr.push_back(keypoints[matches[i].trainIdx].pt);
                frame.vkey_pts_visible_with_last_frame_[i] = 1;
                keypoints_last.push_back(vframe[img_id - 1].vkey_pts_[matches[i].queryIdx].pt);
                if (vframe[img_id - 1].vkey_pts_visible_with_last_frame_[matches[i].queryIdx]) {
                    cur_pts_id_with_last_last.push(matches[i].trainIdx);
                    last_pts_id_com_with_last_last.push(matches[i].queryIdx);
                }
            }
        }

    }

  // save cur feature
    last_keypts=keypoints;
    last_keypts_desc = keypoints_desc.clone();

    if (img_id == 0) {
        frame.rotate_ = cv::Mat::eye(3,3,CV_32F);
        frame.trans_ = cv::Mat::zeros(1,3, CV_32F);
        continue;
    }
    std::cout<<"-- keypoints_curr "<<keypoints_curr.size()<<" keypoints_last "<<keypoints_last.size()<<std::endl;
    cv::Mat E = cv::findEssentialMat(keypoints_curr, keypoints_last, cameraMatrix);
  

    std::cout<<"-- EssentialMat "<<E<<std::endl;

     cv::Mat R, T;
     cv::recoverPose(E, keypoints_curr, keypoints_last, R, T);
     std::cout<<"-- R "<<R<<" T "<<T<<std::endl;

    // compute 3d coordinate
     if (img_id > 1) {
         // search common pts in cur, last and lastlast
         
         for(int i = 0; i < vframe[img_id - 1].vkey_pts_.size(); ++i) {
            int visbile = frame[img_id - 1].vkey_pts_visible_with_last_frame_[i];
            // search common pts
            if (visbile) {

            }
         }
         E = cv::findEssentialMat(keypoints_curr, keypoints_last, cameraMatrix);
         cv::recoverPose(E, keypoints_curr, keypoints_last, R, T);
     }
     if (img_id == 1) {
        frame.rotate_ = R.clone();
        frame.trans_ = T.clone();

        // Create two relative pose
        // P1 = K [  I    |   0  ]
        // P2 = K [R{1,2} | {+-}t]
        cv::Mat proj_ref, proj_cur;
        hconcat(cameraMatrix, Vec3d::zeros(), proj_ref);
        hconcat(cameraMatrix * R, cameraMatrix * T, proj_cur);
        std::vector<cv::Point3f> proj_pts;
        cv::triangulatePoints(proj_ref, proj_cur, keypoints_last, keypoints_curr, proj_pts);
        for(auto pt : &proj_pts) 
           pt /=pt(3);
        continue;
     } else if (img_id > 1) {

     }


    cv::imshow("KeyPoints Image", keypoint_img);
    cv::waitKey(30);

    }

    // 3. end
    // std::cout<<"-- shutdown"<<std::endl;
}
