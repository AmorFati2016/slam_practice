cmake_minimum_required(VERSION 2.6)
project(slam_demo)

SET(CMAKE_CXX_COMPILER "g++")
set(OpenCV_OK opencv_calib3d opencv_core opencv_dnn opencv_features2d opencv_flann opencv_gapi opencv_highgui opencv_imgcodecs opencv_imgproc opencv_ml opencv_objdetect opencv_photo opencv_stitching opencv_video opencv_videoio opencv_aruco opencv_bgsegm opencv_bioinspired opencv_ccalib opencv_datasets opencv_dnn_objdetect opencv_dpm opencv_face opencv_freetype opencv_fuzzy opencv_hfs opencv_img_hash opencv_line_descriptor opencv_optflow opencv_phase_unwrapping opencv_plot opencv_reg opencv_rgbd opencv_saliency opencv_shape opencv_stereo opencv_structured_light opencv_superres opencv_surface_matching opencv_text opencv_tracking opencv_videostab opencv_xfeatures2d opencv_ximgproc opencv_xobjdetect opencv_xphoto)
add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#find_package(Ceres REQUIRED)
#include_directories(${CERES_INCLUDE_DIRS})
include_directories(include/)

# 增加PCL库的依赖
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/Cellar/qt@5/5.15.2_1/lib/cmake/Qt5")
#FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filter)
FIND_PACKAGE( PCL REQUIRED)

#list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

# 添加头文件和库文件
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

add_executable(slam_demo main.cc)
#target_link_libraries(slam_demo ${OpenCV_OK} ${CERES_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(slam_demo ${OpenCV_OK} ${PCL_LIBRARIES})
