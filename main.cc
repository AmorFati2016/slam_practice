
#include <iostream>
#include <fstream>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

cv::Mat last_frame;
cv::Mat last_keypts_desc;
std::vector<cv::KeyPoint> last_keypts;

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

        left_img_path.push_back(left_img_root + "/" + ss.str() + ".png");
        right_img_path.push_back(right_img_root + "/" + ss.str() + ".png");
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
    // data load
    left_img = cv::imread(left_img_path[img_id], 0);
    right_img = cv::imread(right_img_path[img_id], 0);

    cv::Mat left, right;
    cv::fisheye::undistortImage(left_img, left_img, cameraMatrix, distCoeffs, cameraMatrix, left_img.size());
    cv::fisheye::undistortImage(right_img, right_img, cameraMatrix, distCoeffs, cameraMatrix, right_img.size());




    // feature detect
    cv::Ptr<cv::ORB> detector = cv::ORB::create(1000);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat keypoints_desc;
    detector->detectAndCompute(left_img, cv::Mat(), keypoints, keypoints_desc);
 


     // 绘制关键点
    cv::Mat keypoint_img;
    cv::drawKeypoints(left_img, keypoints, keypoint_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

     cv::imshow("KeyPoints Image", keypoint_img);
    cv::waitKey(30);
    continue;
    
    // feature match
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
    std::vector<cv::Point2f> keypoints_curr, keypoints_last;
    if (img_id == 0) {

    } else {
        std::vector<cv::DMatch> matches;
        matcher->match(keypoints_desc, last_keypts_desc, matches);

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
                keypoints_last.push_back(last_keypts[matches[i].queryIdx].pt);
            }
        }

    }

    // // save cur feature
    // last_keypts=keypoints;
    // last_keypts_desc = keypoints_desc.clone();

    // if (img_id == 0) continue;

    // // pose estimate
    // cv::Mat F = cv::findFundamentalMat(keypoints_curr, keypoints_last, cv::FM_RANSAC);
    // std::cout<<"-- F "<<F<<std::endl;
    // // pt3 = F*cv::Mat(pt3);

    // float z = F.at<double>(2,0) *pt3.x + F.at<double>(2,1) *pt3.y + F.at<double>(2,2) *pt3.z;
    // float x = F.at<double>(0,0) *pt3.x + F.at<double>(0,1) *pt3.y + F.at<double>(0,2) *pt3.z;
    // float y = F.at<double>(1,0) *pt3.x + F.at<double>(1,1) *pt3.y + F.at<double>(1,2) *pt3.z;
    // pt3.x = x;
    // pt3.y = y;
    // pt3.z = z;

    // cv::circle(screen, cv::Point2f(x/z, y/z) * 100, 3, cv::Scalar(255));
    // cv::imshow("points", screen);
    }

    // 3. end
    // std::cout<<"-- shutdown"<<std::endl;
}
