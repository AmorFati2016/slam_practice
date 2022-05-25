
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

// g2o
#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/types_slam3d.h"
#include "g2o/types/slam3d/vertex_se3.h"

std::string root_path = "/home/wangpeng04/work/slam/slam_practice/data";


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

void Normalize(std::vector<cv::Point2f> &pts, std::vector<cv::Point2f> &pts_norm, cv::Mat &t) {
    // point number
    int n = pts.size();
    pts_norm.resize(n);

    float meanx = 0;
    float meany = 0;

    for(auto pt : pts) {
        meanx +=pt.x;
        meany +=pt.y;
    } 

    // mean value
    meanx /=n;
    meany /=n;

    // mean dev
    float meandevx = 0;
    float meandevy = 0;

    pts_norm = pts;
    for(auto &pt : pts_norm) {
        pt.x -= meanx;
        pt.y -= meany;

        meandevx +=abs(pt.x);
        meandevy +=abs(pt.y);
    }

    float sx = 1.0 * n / meandevx;
    float sy = 1.0 * n / meandevy;

    for(auto &pt : pts_norm) {
        pt.x *= sx;
        pt.y *= sy;
    }

    t = (cv::Mat_<float>(2,2)<<sx, 0, -1 * meanx * sx, 0, sy, -1 * meany * sy, 0, 0, 1);
}

cv::Mat ComputeFundamentalMatrix(std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2) {
    // x'TFx=0
    // AF = (x'x, x'y, x', y'x, y'y, y', x, y, 1)f=0
    // build A
    int N = pts1.size();
    cv::Mat A(N,9,CV_32F);

    for(int i = 0; i < N; ++i) {
        A.ptr<float>(i)[0] = pts2[i].x * pts1[i].x;
        A.ptr<float>(i)[1] = pts2[i].x * pts1[i].y;
        A.ptr<float>(i)[2] = pts2[i].x;
        A.ptr<float>(i)[3] = pts2[i].y * pts1[i].x;
        A.ptr<float>(i)[4] = pts2[i].y * pts1[i].y;
        A.ptr<float>(i)[5] = pts2[i].y;
        A.ptr<float>(i)[6] = pts1[i].x;
        A.ptr<float>(i)[7] = pts1[i].y;
        A.ptr<float>(i)[8] = 1;
    }

    // SVD
    // The least-squares solution for f is the singular vector corresponding to the smallest singular value of A
    // that is, the last column of V in the SVD A = UDVT .
    cv::Mat u,w,vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    // The matrix F found by solving the set of linear equations will not in general have rank 2
    // and we should take steps to enforce this constraint. A convenient method of doing this is to again use the SVD.
    // F = UDVT be the SVD of F
    // F' = Udiag(r, s, 0)VT minimizes the Frobenius norm of F − F'.
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}

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
// 估计一个运动的大小
double normofTransform( cv::Mat rvec, cv::Mat tvec );
// 检测两个帧，结果定义
enum CHECK_RESULT {NOT_MATCHED=0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME}; 
// 函数声明
CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops=false );

int main(int argc, char* argv[]) {
    
    using cloud_type = pcl::PointCloud<pcl::PointXYZRGBA>;
    cv::Mat camera_matrix = (cv::Mat_<float>(3,3)<<camera_fx,0, camera_cx, 0, camera_fy, camera_cy, 0, 0, 1);

    int start_index = 1;

    // show
    // pcl::visualization::CloudViewer* cloud_viewer(new pcl::visualization::CloudViewer("viewer"));
    FRAME frame_cur, frame_last;
    std::vector<cv::KeyPoint> last_kps;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr last_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>());



    LoadData(start_index, &frame_cur);
    cloud_type::Ptr point_cloud = DepthToPointCloud(frame_cur.img_depth, frame_cur.img_rgb, camera_matrix, camera_factor);
    cv::Mat img_kpt_desc;
    std::vector<cv::KeyPoint> img_kpt;
    FeaturesDection(frame_cur.img_rgb, &img_kpt_desc, &img_kpt);


    frame_last = frame_cur;
    frame_last.img_kpt_desc = img_kpt_desc.clone();
    frame_last.img_kpt.swap(img_kpt);

    *last_point_cloud = *point_cloud;


    /******************************* 
    // 新增:有关g2o的初始化
    *******************************/
    //  define the optimizer
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer globalOptimizer;  // 最后用的就是这个东东
    globalOptimizer.setAlgorithm( solver ); 
    // 不要输出调试信息
    globalOptimizer.setVerbose( false );
    
    // 向globalOptimizer增加第一个顶点
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId( start_index );
    v->setEstimate( Eigen::Isometry3d::Identity() ); //估计为单位矩阵
    v->setFixed( true ); //第一个顶点固定，不用优化
    globalOptimizer.addVertex( v );

    for(auto img_id = start_index + 1; img_id < 30; ++img_id) {


        LoadData(img_id, &frame_cur);
        cloud_type::Ptr point_cloud = DepthToPointCloud(frame_cur.img_depth, frame_cur.img_rgb, camera_matrix, camera_factor);
        cv::Mat img_kpt_desc;
        std::vector<cv::KeyPoint> img_kpt;
        FeaturesDection(frame_cur.img_rgb, &img_kpt_desc, &img_kpt);


        std::cout<<"-- cur "<<img_kpt.size()<<" last "<<frame_last.img_kpt.size()<<std::endl;
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

    //     // 2d - 2d
    //     cv::Mat mask, R2d, T2d;
    //     cv::Mat E = cv::findEssentialMat(keypoints_curr, keypoints_last, camera_matrix, cv::RANSAC, 0.999, 1.0, 1000, mask);
    //     cv::recoverPose(E, keypoints_curr, keypoints_last, camera_matrix, R2d, T2d);
    //     std::cout<<"-- 2d-2d estimate "<<R2d<<" "<<T2d<<std::endl;

    //    // 3d - 3d
    //     pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
    //     icp.setInputSource(last_point_cloud);
    //     icp.setInputTarget(point_cloud);
        
    //     pcl::PointCloud<pcl::PointXYZRGBA> Final;
    //     icp.align(Final);
      
    //     std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    //     icp.getFitnessScore() << std::endl;
    //     std::cout << icp.getFinalTransformation() << std::endl;


        // // point cloud transform
        // pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>());
       
        //  // Create the filtering object
        //  pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
        //  sor.setInputCloud (point_cloud);
        //  sor.setLeafSize (0.3f, 0.3f, 0.3f);
        //  sor.filter (*cloud_filtered);
        //  *point_cloud = *cloud_filtered;

        // //
        // pcl::PointCloud<pcl::PointXYZRGBA>::Ptr new_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
        // pcl::transformPointCloud(*last_point_cloud, *new_point_cloud, transform_1);
        // *point_cloud += *new_point_cloud;
        // *last_point_cloud = *point_cloud;
        
        frame_last = frame_cur;
        frame_last.img_kpt_desc = img_kpt_desc.clone();
        frame_last.img_kpt.swap(img_kpt);
        // frame_last.img_kpt.swap(img_kpt);

        // cloud_viewer->showCloud(last_point_cloud);
        
    
    }

        // pcl::io::savePCDFile("point_cloud.pcd", *last_point_cloud);


    // show

    // pnp


    return 1;
}


double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}


CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops)
{
    static ParameterReader pd;
    static int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    static double max_norm = atof( pd.getData("max_norm").c_str() );
    static double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    static double max_norm_lp = atof( pd.getData("max_norm_lp").c_str() );
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    // 比较f1 和 f2
    RESULT_OF_PNP result = estimateMotion( f1, f2, camera );
    if ( result.inliers < min_inliers ) //inliers不够，放弃该帧
        return NOT_MATCHED;
    // 计算运动范围是否太大
    double norm = normofTransform(result.rvec, result.tvec);
    if ( is_loops == false )
    {
        if ( norm >= max_norm )
            return TOO_FAR_AWAY;   // too far away, may be error
    }
    else
    {
        if ( norm >= max_norm_lp)
            return TOO_FAR_AWAY;
    }

    if ( norm <= keyframe_threshold )
        return TOO_CLOSE;   // too adjacent frame
    // 向g2o中增加这个顶点与上一帧联系的边
    // 顶点部分
    // 顶点只需设定id即可
    if (is_loops == false)
    {
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId( f2.frameID );
        v->setEstimate( Eigen::Isometry3d::Identity() );
        opti.addVertex(v);
    }
    // 边部分
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    // 连接此边的两个顶点id
    edge->setVertex( 0, opti.vertex(f1.frameID ));
    edge->setVertex( 1, opti.vertex(f2.frameID ));
    edge->setRobustKernel( new g2o::RobustKernelHuber() );
    // 信息矩阵
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
    // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
    information(0,0) = information(1,1) = information(2,2) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100;
    // 也可以将角度设大一些，表示对角度的估计更加准确
    edge->setInformation( information );
    // 边的估计即是pnp求解之结果
    Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec );
    // edge->setMeasurement( T );
    edge->setMeasurement( T.inverse() );
    // 将此边加入图中
    opti.addEdge(edge);
    return KEYFRAME;
}